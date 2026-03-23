"""GRPO LoRA training — reward-based optimization for filter generation."""

import re
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, get_peft_model, LoraConfig
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
    text = pred[0]["content"] if isinstance(pred, list) else pred
    # Strip <think>...</think> blocks from thinking-enabled models
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return text


_OP_RE = re.compile(r'(==|!=|>=|<=|>|<|\bNOT\s+IN\b|\bIN\b)', re.I)
_FIELD_RE = re.compile(r'(\w+)\s*(?:==|!=|>=|<=|>|<|\bNOT\s+IN\b|\bIN\b)')
_VALUE_RE = re.compile(r"'([^']*)'|([\d.]+)")


# ── Reward function ───────────────────────────────────────────────────────

def composite_reward(completions, expected, **kwargs):
    """Layered reward that always produces variance across generations.

    Scoring (0.0 – 1.0):
      +0.10  format   — has at least one operator, or correctly says EMPTY
      +0.20  fields   — F1 over field names used in the filter
      +0.40  clauses  — F1 over normalized clauses
      +0.30  exact    — bonus for perfect match
    """
    rewards = []
    for pred, exp in zip(completions, expected):
        text = _extract_text(pred)
        exp_text = exp.strip()
        score = 0.0

        # Handle EMPTY
        if exp_text.upper() == "EMPTY":
            rewards.append(1.0 if text.upper() == "EMPTY" else 0.0)
            continue

        # Level 1: Format — has a valid operator?
        if _OP_RE.search(text):
            score += 0.10

        # Level 2: Field overlap (F1)
        pred_fields = set(_FIELD_RE.findall(text))
        exp_fields = set(_FIELD_RE.findall(exp_text))
        if exp_fields and pred_fields:
            overlap = pred_fields & exp_fields
            p = len(overlap) / len(pred_fields)
            r = len(overlap) / len(exp_fields)
            score += 0.20 * (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        # Level 3: Clause F1
        pred_clauses = _parse(text)
        exp_clauses = _parse(exp_text)
        if pred_clauses and exp_clauses:
            overlap = pred_clauses & exp_clauses
            p = len(overlap) / len(pred_clauses)
            r = len(overlap) / len(exp_clauses)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            score += 0.40 * f1

        # Level 4: Exact match bonus
        if pred_clauses == exp_clauses:
            score += 0.30

        rewards.append(score)
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
    adapter_dir = Path(sft_adapter) if sft_adapter else cfg.adapter_dir / "sft"

    print(f"Loading base model: {cfg.model.name}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)

    if adapter_dir.exists():
        print(f"Loading and merging SFT adapter from {adapter_dir}...")
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model = model.merge_and_unload()
        print("SFT adapter merged.")
    else:
        print(f"WARNING: No SFT adapter at {adapter_dir}, starting from base model")

    # Apply new LoRA for GRPO
    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules="all-linear" if cfg.lora.target_modules == "all-linear" else cfg.lora.target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not cfg.model.enable_thinking:
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
        reward_funcs=[composite_reward],
    )

    trainer.train()

    trainer.save_model(str(grpo_dir))
    tokenizer.save_pretrained(str(grpo_dir))
    print(f"GRPO adapter saved to {grpo_dir}")

    if cfg.wandb.enabled:
        import wandb
        wandb.finish()
