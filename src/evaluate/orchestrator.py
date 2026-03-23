"""Evaluation orchestrator: batched inference + metric computation."""

import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from ..config import Config
from ..data_loader import load_datasets
from ..inference import load_model, QUANTIZATION_MODES
from ..training_utils import strip_thinking_output
from .base import SampleContext, EvaluationResult
from .parsing import extract_schema_columns

from .precision import PrecisionMetric
from .recall import RecallMetric
from .f1 import F1Metric
from .exact_match import ExactMatchMetric
from .field_accuracy import FieldAccuracyMetric
from .hallucination import HallucinationMetric
from .misalignment import MisalignmentMetric
from .latency import LatencyMetric
from .structural_validity import StructuralValidityMetric
from .complexity_accuracy import ComplexityAccuracyMetric
from .operator_accuracy import OperatorAccuracyMetric
from .value_accuracy import ValueAccuracyMetric

METRICS = [
    PrecisionMetric(), RecallMetric(), F1Metric(), ExactMatchMetric(),
    FieldAccuracyMetric(), HallucinationMetric(), MisalignmentMetric(),
    LatencyMetric(), StructuralValidityMetric(), ComplexityAccuracyMetric(),
    OperatorAccuracyMetric(), ValueAccuracyMetric(),
]


def _extract_query(user_content):
    for line in user_content.split("\n"):
        if line.strip().startswith("User query:"):
            return line.split("User query:", 1)[1].strip()
    return ""


def _extract_difficulty(file_path):
    parts = file_path.split("__")
    return parts[1] if len(parts) > 1 else "unknown"


def _get_model_size_mb(model):
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    return param_bytes / (1024 * 1024)


def _run_single(cfg, eval_ds, model, tokenizer, quantization="fp16", verbose=False):
    gen = cfg.generation
    batch_size = gen.eval_batch_size

    prompts, expected_list, schema_columns_list = [], [], []
    schema_names, difficulties, queries = [], [], []

    for sample in eval_ds:
        messages = sample["messages"]
        expected_list.append(messages[2]["content"])
        schema_columns_list.append(extract_schema_columns(messages[1]["content"]))
        queries.append(_extract_query(messages[1]["content"]))
        fp = sample.get("file_path") or ""
        schema_names.append(fp.split("__")[0] if fp else "unknown")
        difficulties.append(_extract_difficulty(fp) if fp else "unknown")
        prompts.append(
            tokenizer.apply_chat_template(
                messages[:2], tokenize=False, add_generation_prompt=True,
                enable_thinking=cfg.model.enable_thinking,
            )
        )

    tokenizer.padding_side = "left"

    predictions, latencies = [], []
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    pbar = tqdm(range(0, len(prompts), batch_size), total=num_batches,
                desc=f"Inference [{quantization}]")

    for batch_start in pbar:
        batch_prompts = prompts[batch_start: batch_start + batch_size]
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(model.device)

        do_sample = gen.temperature > 0
        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=gen.max_new_tokens,
                temperature=gen.temperature if do_sample else None,
                top_p=gen.top_p if do_sample else None,
                top_k=gen.top_k if do_sample else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        batch_time_ms = (time.perf_counter() - t0) * 1000
        per_sample_ms = batch_time_ms / len(batch_prompts)

        for j in range(len(batch_prompts)):
            prompt_len = inputs["attention_mask"][j].sum().item()
            generated = output_ids[j][prompt_len:]
            predicted = tokenizer.decode(generated, skip_special_tokens=True).strip()
            predicted = strip_thinking_output(predicted)
            predictions.append(predicted)
            latencies.append(per_sample_ms)

        pbar.set_postfix(ms_per_sample=f"{per_sample_ms:.0f}")

    if verbose:
        print(f"\n{'=' * 60}\n  PREDICTIONS vs EXPECTED\n{'=' * 60}")
        for i in range(len(predictions)):
            match = "Y" if predictions[i].strip() == expected_list[i].strip() else "N"
            print(f"\n  [{i}] {match} {difficulties[i]}")
            print(f"    Expected:  {expected_list[i][:200]}")
            print(f"    Predicted: {predictions[i][:200]}")

    per_metric_results = {m.name: [] for m in METRICS}
    per_schema_results = defaultdict(lambda: {m.name: [] for m in METRICS})
    per_difficulty_results = defaultdict(lambda: {m.name: [] for m in METRICS})

    for i in tqdm(range(len(predictions)), desc="Computing metrics"):
        ctx = SampleContext(
            schema_columns=schema_columns_list[i],
            schema_name=schema_names[i],
            difficulty=difficulties[i],
            latency_ms=latencies[i],
            query=queries[i],
        )
        for metric in METRICS:
            value = metric.compute_sample(predictions[i], expected_list[i], ctx)
            per_metric_results[metric.name].append(value)
            per_schema_results[schema_names[i]][metric.name].append(value)
            per_difficulty_results[difficulties[i]][metric.name].append(value)

    overall = {}
    for metric in METRICS:
        overall.update(metric.aggregate(per_metric_results[metric.name]))
    overall["model_size_mb"] = _get_model_size_mb(model)

    result = EvaluationResult(
        quantization=quantization, overall=overall,
        per_schema=dict(per_schema_results),
        per_difficulty=dict(per_difficulty_results),
        predictions=predictions,
    )
    return result, expected_list, queries, schema_names, difficulties


def _save_predictions(result, eval_ds, expected_list, queries, schema_names,
                      difficulties, output_path):
    samples = []
    for i, pred in enumerate(result.predictions):
        samples.append({
            "index": i,
            "schema_name": schema_names[i],
            "query_type": difficulties[i],
            "query": queries[i],
            "expected": expected_list[i],
            "predicted": pred,
            "exact_match": pred.strip() == expected_list[i].strip(),
        })

    out = {"model": result.quantization, "overall": result.overall, "samples": samples}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nPredictions saved to {output_path}")


def _print_results(result, num_samples):
    quant = result.quantization
    overall = result.overall

    print(f"\n{'=' * 60}")
    print(f"  RESULTS [{quant.upper()}]")
    print(f"{'=' * 60}")
    print(f"  {'Samples':<30s} {num_samples}")
    print(f"  {'Model Size (MB)':<30s} {overall.get('model_size_mb', 0):.1f}")
    for key, val in overall.items():
        if key == "model_size_mb":
            continue
        if isinstance(val, float):
            fmt = f"{val:.2f}" if val > 1.0 or "ms" in key else f"{val:.3f}"
            print(f"  {key:<30s} {fmt}")

    if len(result.per_schema) > 1:
        print(f"\n{'=' * 60}")
        print(f"  PER-SCHEMA BREAKDOWN [{quant.upper()}]")
        print(f"{'=' * 60}")
        print(f"  {'Schema':<30s} {'N':>5s} {'F1':>7s} {'EM':>7s} {'FA':>7s} {'Prec':>7s} {'Rec':>7s}")
        print("  " + "-" * 74)
        for schema in sorted(result.per_schema.keys()):
            data = result.per_schema[schema]
            n = len(data["f1"])
            f1 = sum(data["f1"]) / n
            em = sum(data["exact_match"]) / n
            fa = sum(data["field_accuracy"]) / n
            p = sum(data["precision"]) / n
            r = sum(data["recall"]) / n
            print(f"  {schema:<30s} {n:>5d} {f1:>7.3f} {em:>7.3f} {fa:>7.3f} {p:>7.3f} {r:>7.3f}")

    if len(result.per_difficulty) > 1:
        print(f"\n{'=' * 60}")
        print(f"  PER-DIFFICULTY BREAKDOWN [{quant.upper()}]")
        print(f"{'=' * 60}")
        print(f"  {'Difficulty':<35s} {'N':>5s} {'F1':>7s} {'EM':>7s} {'FA':>7s} {'SV':>7s}")
        print("  " + "-" * 66)
        for diff in sorted(result.per_difficulty.keys()):
            data = result.per_difficulty[diff]
            n = len(data["f1"])
            f1 = sum(data["f1"]) / n
            em = sum(data["exact_match"]) / n
            fa = sum(data["field_accuracy"]) / n
            sv = sum(data["structural_validity"]) / n
            print(f"  {diff:<35s} {n:>5d} {f1:>7.3f} {em:>7.3f} {fa:>7.3f} {sv:>7.3f}")


def _print_comparison(all_results):
    print(f"\n{'=' * 80}")
    print("  QUANTIZATION COMPARISON")
    print(f"{'=' * 80}")

    quants = [r.quantization.upper() for r in all_results]
    print(f"  {'Metric':<30s}" + "".join(f" {q:>10s}" for q in quants))
    print("  " + "-" * (30 + 11 * len(quants)))

    for key in all_results[0].overall:
        row = f"  {key:<30s}"
        for result in all_results:
            val = result.overall.get(key, 0)
            if isinstance(val, float):
                fmt = f"{val:>10.2f}" if val > 1.0 or "ms" in key else f"{val:>10.3f}"
                row += fmt
            else:
                row += f" {val:>10s}"
        print(row)
    print()


def main(cfg=None, max_samples=None, zero_shot=False, quantizations=None,
         verbose=False, sft_adapter=None, grpo_adapter=None):
    cfg = cfg or Config()
    quantizations = quantizations or ["fp16"]

    _, eval_ds = load_datasets(cfg)
    if max_samples:
        eval_ds = eval_ds.select(range(min(max_samples, len(eval_ds))))

    print(f"\nEval samples: {len(eval_ds)}")
    print(f"Quantizations: {', '.join(quantizations)}")
    print(f"Metrics: {', '.join(m.name for m in METRICS)}\n")

    all_results = []

    for quant in quantizations:
        print(f"\n>>> Loading model [{quant.upper()}] ...")
        model, tokenizer = load_model(
            cfg, zero_shot=zero_shot, quantization=quant,
            sft_adapter=sft_adapter, grpo_adapter=grpo_adapter,
        )

        result, expected_list, queries, schema_names, difficulties = _run_single(
            cfg, eval_ds, model, tokenizer, quantization=quant, verbose=verbose,
        )
        all_results.append(result)
        _print_results(result, len(eval_ds))

        if zero_shot:
            save_dir = Path(cfg.paths.output_dir) / cfg.model.name.split("/")[-1] / "zero_shot" / "metrics"
            eval_name = f"eval_{quant}.json"
        else:
            save_dir = cfg.adapter_dir / "metrics"
            stage = "grpo" if grpo_adapter else "sft"
            eval_name = f"eval_{stage}_{quant}.json"
        _save_predictions(result, eval_ds, expected_list, queries, schema_names,
                          difficulties, str(save_dir / eval_name))

        if len(quantizations) > 1:
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

    if len(all_results) > 1:
        _print_comparison(all_results)

    return all_results[0] if len(all_results) == 1 else all_results
