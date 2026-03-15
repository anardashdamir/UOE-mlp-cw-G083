"""Orchestrator: loads model, runs batched inference, computes all metrics.

This is the central entry point for evaluation. It:
1. Loads the model and tokenizer (with optional quantization)
2. Prepares all eval samples
3. Runs batched inference with timing
4. Passes each (predicted, expected) pair to every registered metric
5. Aggregates and prints results with per-schema and per-difficulty breakdowns
6. Supports running multiple quantization modes in one invocation for comparison
"""

import gc
import time
from collections import defaultdict

import torch
from tqdm import tqdm

from ..config import Config
from ..data_loader import load_datasets
from ..inference import load_model, QUANTIZATION_MODES
from ..training_utils import strip_thinking_output
from .base import SampleContext, EvaluationResult
from .parsing import extract_schema_columns

# ── Register all metrics here ───────────────────────────────────────────────
# To add a new metric, import it and append to METRICS.
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
    PrecisionMetric(),
    RecallMetric(),
    F1Metric(),
    ExactMatchMetric(),
    FieldAccuracyMetric(),
    HallucinationMetric(),
    MisalignmentMetric(),
    LatencyMetric(),
    StructuralValidityMetric(),
    ComplexityAccuracyMetric(),
    OperatorAccuracyMetric(),
    ValueAccuracyMetric(),
]


def _extract_schema_name(user_content: str) -> str:
    """Extract a schema identifier from the user message."""
    lines = user_content.split("\n")
    for line in lines:
        if line.startswith("<schema>"):
            continue
        if ":" in line:
            return line.split(":")[0].strip()
    return "unknown"


def _extract_difficulty(file_path: str) -> str:
    """Extract difficulty/type from file_path (second __ segment)."""
    parts = file_path.split("__")
    return parts[1] if len(parts) > 1 else "unknown"


def _get_model_size_mb(model) -> float:
    """Estimate model memory footprint in MB."""
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    return param_bytes / (1024 * 1024)


def _run_single(
    cfg: Config,
    eval_ds,
    model,
    tokenizer,
    quantization: str = "fp16",
    verbose: bool = False,
) -> dict:
    """Run evaluation for a single model/quantization configuration."""
    gen = cfg.generation
    batch_size = gen.eval_batch_size

    # ── Prepare all prompts and metadata ────────────────────────────────────
    prompts = []
    expected_list = []
    schema_columns_list = []
    schema_names = []
    difficulties = []

    for sample in eval_ds:
        messages = sample["messages"]
        expected_list.append(messages[2]["content"])
        schema_columns_list.append(extract_schema_columns(messages[1]["content"]))
        schema_names.append(_extract_schema_name(messages[1]["content"]))
        fp = sample.get("file_path") or ""
        difficulties.append(_extract_difficulty(fp) if fp else "unknown")
        prompts.append(
            tokenizer.apply_chat_template(
                messages[:2], tokenize=False, add_generation_prompt=True,
                enable_thinking=cfg.model.enable_thinking,
            )
        )

    tokenizer.padding_side = "left"

    # ── Batched inference ───────────────────────────────────────────────────
    predictions = []
    latencies = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    pbar = tqdm(
        range(0, len(prompts), batch_size),
        total=num_batches,
        desc=f"Inference [{quantization}]",
    )

    for batch_start in pbar:
        batch_prompts = prompts[batch_start: batch_start + batch_size]
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=cfg.training.max_seq_length,
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
            if cfg.model.enable_thinking:
                predicted = strip_thinking_output(predicted)
            predictions.append(predicted)
            latencies.append(per_sample_ms)

        pbar.set_postfix(ms_per_sample=f"{per_sample_ms:.0f}")

    # ── Debug: print predictions vs expected ─────────────────────────────
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  PREDICTIONS vs EXPECTED")
        print(f"{'=' * 60}")
        for i in range(len(predictions)):
            match = "✓" if predictions[i].strip() == expected_list[i].strip() else "✗"
            print(f"\n  [{i}] {match} {difficulties[i]}")
            print(f"    Expected:  {expected_list[i][:200]}")
            print(f"    Predicted: {predictions[i][:200]}")

    # ── Compute all metrics ─────────────────────────────────────────────────
    per_metric_results = {m.name: [] for m in METRICS}
    per_schema_results = defaultdict(lambda: {m.name: [] for m in METRICS})
    per_difficulty_results = defaultdict(lambda: {m.name: [] for m in METRICS})

    for i in tqdm(range(len(predictions)), desc="Computing metrics"):
        ctx = SampleContext(
            schema_columns=schema_columns_list[i],
            schema_name=schema_names[i],
            difficulty=difficulties[i],
            latency_ms=latencies[i],
        )
        for metric in METRICS:
            value = metric.compute_sample(predictions[i], expected_list[i], ctx)
            per_metric_results[metric.name].append(value)
            per_schema_results[schema_names[i]][metric.name].append(value)
            per_difficulty_results[difficulties[i]][metric.name].append(value)

    # ── Aggregate ───────────────────────────────────────────────────────────
    overall = {}
    for metric in METRICS:
        agg = metric.aggregate(per_metric_results[metric.name])
        overall.update(agg)

    # Add model size
    overall["model_size_mb"] = _get_model_size_mb(model)

    return EvaluationResult(
        quantization=quantization,
        overall=overall,
        per_schema=dict(per_schema_results),
        per_difficulty=dict(per_difficulty_results),
        predictions=predictions,
    )


def _print_results(result: EvaluationResult, num_samples: int):
    """Print results for a single quantization run."""
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
            if val > 1.0 or "ms" in key or "hallucination" in key or "misaligned" in key:
                print(f"  {key:<30s} {val:.2f}")
            else:
                print(f"  {key:<30s} {val:.3f}")

    if len(result.per_schema) > 1:
        print(f"\n{'=' * 60}")
        print(f"  PER-SCHEMA BREAKDOWN [{quant.upper()}]")
        print(f"{'=' * 60}")
        header = f"  {'Schema':<30s} {'N':>5s} {'F1':>7s} {'EM':>7s} {'FA':>7s} {'Prec':>7s} {'Rec':>7s}"
        print(header)
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
        header = f"  {'Difficulty':<35s} {'N':>5s} {'F1':>7s} {'EM':>7s} {'FA':>7s} {'SV':>7s}"
        print(header)
        print("  " + "-" * 66)
        for diff in sorted(result.per_difficulty.keys()):
            data = result.per_difficulty[diff]
            n = len(data["f1"])
            f1 = sum(data["f1"]) / n
            em = sum(data["exact_match"]) / n
            fa = sum(data["field_accuracy"]) / n
            sv = sum(data["structural_validity"]) / n
            print(f"  {diff:<35s} {n:>5d} {f1:>7.3f} {em:>7.3f} {fa:>7.3f} {sv:>7.3f}")


def _print_comparison(all_results: list[EvaluationResult]):
    """Print side-by-side comparison of multiple quantization runs."""
    print(f"\n{'=' * 80}")
    print("  QUANTIZATION COMPARISON")
    print(f"{'=' * 80}")

    quants = [r.quantization.upper() for r in all_results]
    header = f"  {'Metric':<30s}" + "".join(f" {q:>10s}" for q in quants)
    print(header)
    print("  " + "-" * (30 + 11 * len(quants)))

    keys = list(all_results[0].overall.keys())
    for key in keys:
        row = f"  {key:<30s}"
        for result in all_results:
            val = result.overall.get(key, 0)
            if isinstance(val, float):
                if val > 1.0 or "ms" in key or "hallucination" in key or "misaligned" in key:
                    row += f" {val:>10.2f}"
                else:
                    row += f" {val:>10.3f}"
            else:
                row += f" {val:>10s}"
        print(row)
    print()


def main(
    cfg: Config = None,
    max_samples: int = None,
    zero_shot: bool = False,
    quantizations: list[str] | None = None,
    verbose: bool = False,
) -> EvaluationResult | list[EvaluationResult]:
    """Run evaluation, optionally across multiple quantization modes.

    Args:
        cfg: Config object.
        max_samples: Limit eval to N samples.
        zero_shot: Evaluate base model without adapter.
        quantizations: List of quantization modes (e.g., ["fp16", "int8", "int4"]).
                       Defaults to ["fp16"].

    Returns:
        Single EvaluationResult or list of EvaluationResult if multiple quantizations.
    """
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
        model, tokenizer = load_model(cfg, zero_shot=zero_shot, quantization=quant)

        result = _run_single(cfg, eval_ds, model, tokenizer, quantization=quant, verbose=verbose)
        all_results.append(result)

        _print_results(result, len(eval_ds))

        # Free GPU memory before loading next quantization
        if len(quantizations) > 1:
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

    # Print comparison table if multiple quantizations
    if len(all_results) > 1:
        _print_comparison(all_results)

    if len(all_results) == 1:
        return all_results[0]
    return all_results
