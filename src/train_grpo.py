"""GRPO LoRA training using Unsloth — reward-based optimization for filter generation."""

import re
from pathlib import Path

from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

from .config import Config
from .data_loader import load_grpo_dataset
from .evaluate.parsing import normalize_clause, split_top_level_and
from .training_utils import build_run_name, disable_thinking, setup_logging


# ── Helpers ───────────────────────────────────────────────────────────────

def _parse(text):
    """Parse filter into normalized clause set."""
    text = text.strip()
    if text.upper() == "EMPTY":
        return {"EMPTY"}
    return {normalize_clause(c) for c in split_top_level_and(text)}


def _extract_text(pred):
    return pred[0]["content"] if isinstance(pred, list) else pred


_OP_RE = re.compile(r'(==|!=|>=|<=|>|<|\bNOT\s+IN\b|\bIN\b)', re.I)
_FIELD_RE = re.compile(r'(\w+)\s*(?:==|!=|>=|<=|>|<|\bNOT\s+IN\b|\bIN\b)')
_VALUE_RE = re.compile(r"'([^']*)'|([\d.]+)")


# ── Reward functions (each returns 0.0 - 1.0) ────────────────────────────

def exact_match_reward(completions, expected, **kwargs):
    """Binary: 1.0 if filter matches exactly, else 0.0."""
    rewards = []
    for pred, exp in zip(completions, expected):
        text = _extract_text(pred)
        rewards.append(1.0 if _parse(text) == _parse(exp.strip()) else 0.0)
    return rewards


def syntax_reward(completions, expected, **kwargs):
    """Valid structure: has operators, balanced parens and quotes."""
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
    """Proportion of correctly used field names."""
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


def clause_f1_reward(completions, expected, **kwargs):
    """F1 score over normalized clauses."""
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


def hallucination_penalty(completions, expected, **kwargs):
    """Penalize extra clauses not in expected. Returns 0.0-1.0 (1.0 = no hallucination)."""
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


# ── Main ──────────────────────────────────────────────────────────────────

def main(cfg: Config = None, sft_adapter: str | None = None):
    cfg = cfg or Config()
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    grpo = cfg.grpo

    run_name = cfg.wandb.run_name or build_run_name(cfg, prefix="grpo")
    print(f"Run: {run_name}")
    report_to, logging_dir = setup_logging(cfg, run_name)

    dataset = load_grpo_dataset(cfg)
    print(f"GRPO samples: {len(dataset)}")

    # Load SFT model as starting point
    adapter_dir = Path(sft_adapter) if sft_adapter else cfg.adapter_dir

    if adapter_dir.exists():
        print(f"Loading SFT adapter from {adapter_dir}, merging into base model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            cfg.model.name,
            max_seq_length=cfg.training.max_seq_length,
            load_in_4bit=False,
            dtype=None,
        )
        # Load and merge SFT adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model = model.merge_and_unload()
        print("SFT adapter merged.")
    else:
        print(f"WARNING: No SFT adapter at {adapter_dir}, starting from base model")
        model, tokenizer = FastLanguageModel.from_pretrained(
            cfg.model.name,
            max_seq_length=cfg.training.max_seq_length,
            load_in_4bit=False,
            dtype=None,
        )

    # Apply new LoRA for GRPO
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules if isinstance(cfg.lora.target_modules, list) else "all-linear",
        use_gradient_checkpointing="unsloth" if cfg.training.gradient_checkpointing else False,
        random_state=42,
    )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not cfg.model.enable_thinking:
        disable_thinking(tokenizer)

    training_args = GRPOConfig(
        output_dir=str(cfg.paths.output_dir / "grpo"),
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
        report_to=report_to,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[exact_match_reward, syntax_reward, field_reward, clause_f1_reward, hallucination_penalty],
        reward_weights=[5.0, 1.0, 1.0, 2.0, 1.0],
    )

    trainer.train()

    grpo_adapter_dir = cfg.paths.output_dir / "grpo_adapter"
    trainer.save_model(str(grpo_adapter_dir))
    tokenizer.save_pretrained(str(grpo_adapter_dir))
    print(f"GRPO adapter saved to {grpo_adapter_dir}")

    if cfg.wandb.enabled:
        import wandb
        wandb.finish()
