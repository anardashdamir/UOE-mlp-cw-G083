"""GRPO LoRA training — reward-based optimization for filter generation."""

import torch
from peft import LoraConfig, TaskType
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not cfg.model.enable_thinking:
        disable_thinking(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
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
        logging_steps=10,
        bf16=True,
        report_to=report_to,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
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
