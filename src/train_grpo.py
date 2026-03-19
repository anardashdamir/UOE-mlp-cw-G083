"""GRPO LoRA training using Unsloth — reward-based optimization for filter generation."""

import re
from pathlib import Path

from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

from .config import Config
from .data_loader import load_grpo_dataset
from .evaluate.parsing import normalize_clause, split_top_level_and
from .training_utils import build_run_name, disable_thinking, setup_logging


# ── Reward function ───────────────────────────────────────────────────────

def _parse(text):
    """Parse filter into normalized clause set."""
    text = text.strip()
    if text.upper() == "EMPTY":
        return {"EMPTY"}
    return {normalize_clause(c) for c in split_top_level_and(text)}


def combined_reward(completions, expected, **kwargs):
    """Multi-signal reward: exact match=1.0, partial credit capped at 0.8, hallucination penalty."""
    rewards = []
    for pred, exp in zip(completions, expected):
        text = pred[0]["content"] if isinstance(pred, list) else pred
        pred_clauses = _parse(text)
        exp_clauses = _parse(exp.strip())

        if pred_clauses == exp_clauses:
            rewards.append(1.0)
            continue

        score = 0.0

        # Clause F1 (0.0 - 0.5)
        if pred_clauses and exp_clauses:
            overlap = pred_clauses & exp_clauses
            prec = len(overlap) / len(pred_clauses)
            rec = len(overlap) / len(exp_clauses)
            if prec + rec > 0:
                score += 2 * prec * rec / (prec + rec) * 0.5

        # Correct clause count (0.1)
        if len(pred_clauses) == len(exp_clauses):
            score += 0.1

        # Valid syntax (0.1)
        has_op = bool(re.search(r'(==|!=|>=|<=|>|<|\bNOT\s+IN\b|\bIN\b)', text, re.I))
        balanced = text.count('(') == text.count(')') and text.count("'") % 2 == 0
        if has_op and balanced:
            score += 0.1

        # Hallucination penalty (-0.3)
        if pred_clauses and exp_clauses:
            extra = len(pred_clauses - exp_clauses)
            if extra > 0:
                score -= (extra / len(pred_clauses)) * 0.3

        rewards.append(max(min(score, 0.8), 0.0))
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
        reward_funcs=[combined_reward],
    )

    trainer.train()

    grpo_adapter_dir = cfg.paths.output_dir / "grpo_adapter"
    trainer.save_model(str(grpo_adapter_dir))
    tokenizer.save_pretrained(str(grpo_adapter_dir))
    print(f"GRPO adapter saved to {grpo_adapter_dir}")

    if cfg.wandb.enabled:
        import wandb
        wandb.finish()
