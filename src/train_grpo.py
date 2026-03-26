"""GRPO training with reward-based optimization."""

import re
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, get_peft_model, LoraConfig
from trl import GRPOConfig, GRPOTrainer

from .config import Config
from .data_loader import load_grpo_dataset
from .evaluate.parsing import normalize_clause, split_top_level_and
from .training_utils import build_run_name, disable_thinking, enable_thinking, setup_logging


def _parse(text):
    text = text.strip()
    if text.upper() == "EMPTY":
        return {"EMPTY"}
    return {normalize_clause(c) for c in split_top_level_and(text)}


def _extract_text(pred):
    text = pred[0]["content"] if isinstance(pred, list) else pred
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if '<think>' in text:
        text = text.split('</think>')[-1].strip() if '</think>' in text else ''
    return text


_OP_RE = re.compile(r'(==|!=|>=|<=|>|<|\bNOT\s+IN\b|\bIN\b)', re.I)
_FIELD_RE = re.compile(r'(\w+)\s*(?:==|!=|>=|<=|>|<|\bNOT\s+IN\b|\bIN\b)')
_VALUE_RE = re.compile(r"'([^']*)'|([\d.]+)")



def f1_with_em_bonus(completions, expected, **kwargs):
    """Clause F1 (0.0-0.7) + exact match bonus (0.3). Smooth signal with EM incentive."""
    rewards = []
    for pred, exp in zip(completions, expected):
        try:
            text = _extract_text(pred)
            pred_clauses = _parse(text)
            exp_clauses = _parse(exp.strip())

            if pred_clauses == exp_clauses:
                rewards.append(1.0)
                continue
            if not pred_clauses or not exp_clauses:
                rewards.append(0.0)
                continue

            overlap = pred_clauses & exp_clauses
            prec = len(overlap) / len(pred_clauses)
            rec = len(overlap) / len(exp_clauses)
            f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
            rewards.append(0.7 * f1)
        except Exception:
            rewards.append(0.0)
    return rewards


def exact_match_reward(completions, expected, **kwargs):
    """1.0 if predicted filter matches expected exactly, else 0.0."""
    rewards = []
    for pred, exp in zip(completions, expected):
        try:
            text = _extract_text(pred)
            rewards.append(1.0 if _parse(text) == _parse(exp.strip()) else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards



def composite_reward(completions, expected, **kwargs):
    """Partial credit: +0.1 format, +0.2 fields, +0.4 clause F1, +0.3 exact."""
    rewards = []
    for pred, exp in zip(completions, expected):
        text = _extract_text(pred)
        exp_text = exp.strip()
        score = 0.0

        if exp_text.upper() == "EMPTY":
            rewards.append(1.0 if text.upper() == "EMPTY" else 0.0)
            continue

        if _OP_RE.search(text):
            score += 0.10

        pred_fields = set(_FIELD_RE.findall(text))
        exp_fields = set(_FIELD_RE.findall(exp_text))
        if exp_fields and pred_fields:
            overlap = pred_fields & exp_fields
            p = len(overlap) / len(pred_fields)
            r = len(overlap) / len(exp_fields)
            score += 0.20 * (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        pred_clauses = _parse(text)
        exp_clauses = _parse(exp_text)
        if pred_clauses and exp_clauses:
            overlap = pred_clauses & exp_clauses
            p = len(overlap) / len(pred_clauses)
            r = len(overlap) / len(exp_clauses)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            score += 0.40 * f1

        if pred_clauses == exp_clauses:
            score += 0.30

        rewards.append(score)
    return rewards


def clause_f1_reward(completions, expected, **kwargs):
    rewards = []
    for pred, exp in zip(completions, expected):
        text = _extract_text(pred)
        pred_clauses = _parse(text)
        exp_clauses = _parse(exp.strip())
        if pred_clauses == exp_clauses:
            rewards.append(1.0)
            continue
        if not pred_clauses or not exp_clauses:
            rewards.append(0.0)
            continue
        overlap = pred_clauses & exp_clauses
        prec = len(overlap) / len(pred_clauses)
        rec = len(overlap) / len(exp_clauses)
        rewards.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)
    return rewards


def syntax_reward(completions, expected, **kwargs):
    rewards = []
    for pred, exp in zip(completions, expected):
        text = _extract_text(pred).strip()
        if text.upper() == "EMPTY":
            rewards.append(1.0 if exp.strip().upper() == "EMPTY" else 0.0)
            continue
        has_op = bool(_OP_RE.search(text))
        balanced_parens = text.count('(') == text.count(')')
        balanced_quotes = text.count("'") % 2 == 0
        score = (0.4 * has_op) + (0.3 * balanced_parens) + (0.3 * balanced_quotes)
        rewards.append(score)
    return rewards


def field_reward(completions, expected, **kwargs):
    rewards = []
    for pred, exp in zip(completions, expected):
        text = _extract_text(pred)
        pred_fields = set(_FIELD_RE.findall(text))
        exp_fields = set(_FIELD_RE.findall(exp.strip()))
        if not exp_fields:
            rewards.append(1.0 if not pred_fields else 0.0)
            continue
        if not pred_fields:
            rewards.append(0.0)
            continue
        overlap = pred_fields & exp_fields
        prec = len(overlap) / len(pred_fields)
        rec = len(overlap) / len(exp_fields)
        rewards.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)
    return rewards


def hallucination_penalty(completions, expected, **kwargs):
    rewards = []
    for pred, exp in zip(completions, expected):
        text = _extract_text(pred)
        pred_clauses = _parse(text)
        exp_clauses = _parse(exp.strip())
        if not pred_clauses or pred_clauses == exp_clauses:
            rewards.append(1.0)
            continue
        extra = len(pred_clauses - exp_clauses)
        rewards.append(max(1.0 - (extra / len(pred_clauses)), 0.0))
    return rewards


def main(cfg: Config = None, sft_adapter: str | None = None):
    cfg = cfg or Config()
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    grpo = cfg.grpo

    run_name = cfg.wandb.run_name or build_run_name(cfg, prefix="grpo")
    print(f"Run: {run_name}")
    report_to, logging_dir = setup_logging(cfg, run_name)

    dataset = load_grpo_dataset(cfg)
    print(f"GRPO samples: {len(dataset)}")

    adapter_dir = Path(sft_adapter) if sft_adapter else cfg.adapter_dir / "sft"

    print(f"Loading base model: {cfg.model.name}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, torch_dtype="auto", device_map="auto", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)

    if adapter_dir.exists():
        print(f"Merging SFT adapter from {adapter_dir}")
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model = model.merge_and_unload()
    else:
        print(f"WARNING: no SFT adapter at {adapter_dir}, using base model")

    lora_config = LoraConfig(
        r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
        target_modules="all-linear" if cfg.lora.target_modules == "all-linear" else cfg.lora.target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.model.enable_thinking:
        enable_thinking(tokenizer)
    else:
        disable_thinking(tokenizer)

    grpo_dir = cfg.adapter_dir / "grpo"
    training_args = GRPOConfig(
        output_dir=str(grpo_dir),
        run_name=run_name,
        logging_dir=logging_dir,
        num_train_epochs=grpo.num_epochs,
        per_device_train_batch_size=grpo.batch_size,
        gradient_accumulation_steps=grpo.gradient_accumulation_steps,
        learning_rate=grpo.learning_rate,
        max_completion_length=grpo.max_completion_length,
        num_generations=grpo.num_generations,
        beta=grpo.beta,
        max_steps=grpo.max_steps,
        logging_steps=1,
        bf16=True,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        report_to=report_to,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[f1_with_em_bonus],
    )

    trainer.train()
    trainer.save_model(str(grpo_dir))
    tokenizer.save_pretrained(str(grpo_dir))
    print(f"GRPO adapter saved to {grpo_dir}")

    if cfg.wandb.enabled:
        import wandb
        wandb.finish()
