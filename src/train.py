"""SFT LoRA fine-tuning using Unsloth."""

from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

from .config import Config
from .data_loader import load_datasets
from .training_utils import build_run_name, disable_thinking, enable_thinking, setup_logging


def main(cfg: Config = None):
    cfg = cfg or Config()
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)

    run_name = cfg.wandb.run_name or build_run_name(cfg)
    print(f"Run: {run_name}")
    report_to, logging_dir = setup_logging(cfg, run_name)

    train_ds, eval_ds = load_datasets(cfg)
    print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        cfg.model.name,
        max_seq_length=cfg.training.max_seq_length,
        load_in_4bit=cfg.training.use_qlora,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules if isinstance(cfg.lora.target_modules, list) else "all-linear",
        use_gradient_checkpointing="unsloth" if cfg.training.gradient_checkpointing else False,
        random_state=42,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.model.enable_thinking:
        enable_thinking(tokenizer)
    else:
        disable_thinking(tokenizer)

    def apply_chat_template(examples):
        return {"text": [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in examples["messages"]
        ]}

    train_ds = train_ds.map(apply_chat_template, batched=True, remove_columns=["messages", "file_path"])
    eval_ds = eval_ds.map(apply_chat_template, batched=True, remove_columns=["messages", "file_path"])

    training_args = SFTConfig(
        output_dir=str(cfg.paths.output_dir),
        run_name=run_name,
        logging_dir=logging_dir,
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.generation.eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_steps=cfg.training.warmup_steps,
        max_seq_length=cfg.training.max_seq_length,
        max_steps=cfg.training.max_steps,
        eval_strategy="steps",
        eval_steps=0.1,
        save_strategy="steps",
        save_steps=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=cfg.training.logging_steps,
        logging_first_step=True,
        bf16=True,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        report_to=report_to,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        dataset_text_field="text",
    )

    print("Running baseline evaluation...")
    baseline = trainer.evaluate()
    print(f"Baseline eval_loss: {baseline['eval_loss']:.4f}")

    trainer.train()
    trainer.save_model(str(cfg.adapter_dir))
    tokenizer.save_pretrained(str(cfg.adapter_dir))
    print(f"Adapter saved to {cfg.adapter_dir}")

    if cfg.wandb.enabled:
        import wandb
        wandb.finish()
