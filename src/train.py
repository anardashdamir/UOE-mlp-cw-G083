"""LoRA fine-tuning of Qwen2.5-0.5B-Instruct using SFTTrainer."""

from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from .config import Config
from .data_loader import load_datasets


def main(cfg: Config = None):
    cfg = cfg or Config()
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)

    # wandb setup
    report_to = "none"
    if cfg.wandb.enabled:
        import wandb

        if cfg.wandb.wandb_api_key:
            wandb.login(key=cfg.wandb.wandb_api_key)

        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=cfg.model_dump(exclude={"paths", "wandb"}),
        )
        report_to = "wandb"

    train_ds, eval_ds = load_datasets(cfg)
    print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Disable Qwen3 thinking mode — replace <think>\n with empty string in chat template
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template and "<think>" in tokenizer.chat_template:
        tokenizer.chat_template = tokenizer.chat_template.replace(
            "{{- '<think>\\n' }}", "{{- '' }}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
    )

    training_args = SFTConfig(
        output_dir=str(cfg.paths.output_dir),
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        max_length=cfg.training.max_seq_length,
        max_steps=cfg.training.max_steps,
        eval_strategy="steps",
        eval_steps=0.2,
        save_strategy="epoch",
        logging_steps=50,
        bf16=True,
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

    trainer.train()
    trainer.save_model(str(cfg.adapter_dir))
    tokenizer.save_pretrained(str(cfg.adapter_dir))
    print(f"Adapter saved to {cfg.adapter_dir}")

    if cfg.wandb.enabled:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
