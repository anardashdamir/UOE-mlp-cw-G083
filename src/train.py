"""SFT LoRA fine-tuning with TensorBoard / W&B logging."""

import logging

import torch
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from .config import Config
from .data_loader import load_datasets
from .training_utils import (
    build_run_name,
    disable_thinking,
    enable_thinking,
    setup_logging,
)

logger = logging.getLogger(__name__)


def main(cfg: Config = None):
    cfg = cfg or Config()
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)

    run_name = cfg.wandb.run_name or build_run_name(cfg)
    print(f"Run: {run_name}")

    report_to, logging_dir = setup_logging(cfg, run_name)

    print("Loading datasets...")
    train_ds, eval_ds = load_datasets(cfg)
    print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    print(f"Loading tokenizer: {cfg.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.model.enable_thinking:
        enable_thinking(tokenizer)
    else:
        disable_thinking(tokenizer)

    print(f"Loading model: {cfg.model.name} (QLoRA={cfg.training.use_qlora})")
    load_kwargs = {"trust_remote_code": True}
    if cfg.training.use_qlora:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, **load_kwargs)
    print("Model loaded.")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
    )

    training_args = SFTConfig(
        output_dir=str(cfg.paths.output_dir),
        run_name=run_name,
        logging_dir=logging_dir,
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_steps=cfg.training.warmup_steps,
        max_length=cfg.training.max_seq_length,
        max_steps=cfg.training.max_steps,
        eval_strategy="steps",
        eval_steps=0.1,
        save_strategy="steps",
        save_steps=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=cfg.training.logging_steps,
        bf16=True,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        report_to=report_to,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # Evaluate before training to get baseline metrics
    print("Running baseline evaluation (before training)...")
    baseline = trainer.evaluate()
    print(f"Baseline eval_loss: {baseline['eval_loss']:.4f}")

    trainer.train()
    trainer.save_model(str(cfg.adapter_dir))
    tokenizer.save_pretrained(str(cfg.adapter_dir))
    print(f"Adapter saved to {cfg.adapter_dir}")

    if cfg.wandb.enabled:
        import wandb

        wandb.finish()
