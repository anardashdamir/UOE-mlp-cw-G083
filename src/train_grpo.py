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

def exact_match_reward(completions, expected, **kwargs):
    rewards = []
    for pred, exp in zip(completions, expected):
        text = pred[0]["content"] if isinstance(pred, list) else pred
        pred_clauses = {normalize_clause(c) for c in split_top_level_and(text.strip())}
        exp_clauses = {normalize_clause(c) for c in split_top_level_and(exp.strip())}
        rewards.append(1.0 if pred_clauses == exp_clauses else 0.0)
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
        reward_funcs=[exact_match_reward],
    )

    trainer.train()

    grpo_adapter_dir = cfg.paths.output_dir / "grpo_adapter"
    trainer.save_model(str(grpo_adapter_dir))
    tokenizer.save_pretrained(str(grpo_adapter_dir))
    print(f"GRPO adapter saved to {grpo_adapter_dir}")

    if cfg.wandb.enabled:
        import wandb
        wandb.finish()
