"""GRPO LoRA training — reward-based optimization for filter generation."""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from .config import Config
from .data_loader import load_grpo_dataset
from .evaluate.parsing import normalize_clause, split_top_level_and
from .training_utils import build_run_name, disable_thinking, setup_logging


# ── Reward functions ────────────────────────────────────────────────────────

def _parse(text):
    """Parse filter into normalized clause set."""
    text = text.strip()
    if text.upper() == "EMPTY":
        return {"EMPTY"}
    return {normalize_clause(c) for c in split_top_level_and(text)}


def combined_reward(completions, expected, **kwargs):
    """Multi-signal reward: exact match dominant, partial credit for structure.

    Scoring (max 1.0):
      - Exact match:        1.0 (overrides everything)
      - Clause F1:          0.0 - 0.6 (proportion of correct clauses)
      - Correct clause count: 0.1 (same number of conditions)
      - Valid syntax:        0.1 (parseable, has operators)

    Partial rewards are capped at 0.8 to maintain strong incentive for exact match.
    """
    import re
    rewards = []
    for pred, exp in zip(completions, expected):
        text = pred[0]["content"] if isinstance(pred, list) else pred
        pred_clauses = _parse(text)
        exp_clauses = _parse(exp.strip())

        # Exact match → full reward
        if pred_clauses == exp_clauses:
            rewards.append(1.0)
            continue

        score = 0.0

        # Clause-level F1 (0.0 - 0.6)
        if pred_clauses and exp_clauses:
            overlap = pred_clauses & exp_clauses
            precision = len(overlap) / len(pred_clauses) if pred_clauses else 0
            recall = len(overlap) / len(exp_clauses) if exp_clauses else 0
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                score += f1 * 0.6

        # Correct number of clauses (0.1)
        if len(pred_clauses) == len(exp_clauses):
            score += 0.1

        # Valid syntax (0.1)
        text_clean = text.strip()
        has_operator = bool(re.search(r'(==|!=|>=|<=|>|<|CONTAINS|IN\s*\[)', text_clean, re.IGNORECASE))
        balanced_parens = text_clean.count('(') == text_clean.count(')')
        balanced_quotes = text_clean.count("'") % 2 == 0
        if has_operator and balanced_parens and balanced_quotes:
            score += 0.1

        rewards.append(min(score, 0.8))  # Cap partial at 0.8
    return rewards


# ── Main ────────────────────────────────────────────────────────────────────

def main(cfg: Config = None):
    cfg = cfg or Config()
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    grpo = cfg.grpo

    run_name = cfg.wandb.run_name or build_run_name(cfg, prefix="grpo")
    print(f"Run: {run_name}")

    report_to, logging_dir = setup_logging(cfg, run_name)

    dataset = load_grpo_dataset(cfg)
    print(f"GRPO samples: {len(dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not cfg.model.enable_thinking:
        disable_thinking(tokenizer)

    # Load SFT adapter as starting point
    adapter_dir = cfg.adapter_dir
    if adapter_dir.exists():
        print(f"Loading SFT adapter from {adapter_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model = model.merge_and_unload()
        print("SFT adapter merged into base model")
    else:
        print(f"WARNING: No SFT adapter found at {adapter_dir}, starting from base model")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )

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
