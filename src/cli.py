"""Typer CLI for AutoFilter."""

import json
from pathlib import Path

import typer

from .config import Config

app = typer.Typer(
    name="autofilter",
    help="Convert natural-language queries into structured filter expressions.",
    add_completion=False,
)

_config_option = typer.Option(None, "--config", "-c", help="Path to config.yaml.")


@app.command()
def sft(
    config: Path | None = _config_option,
    epochs: int | None = typer.Option(None, "--epochs", "-e"),
    batch_size: int | None = typer.Option(None, "--batch-size", "-b"),
    lr: float | None = typer.Option(None, "--lr"),
    lora_r: int | None = typer.Option(None, "--lora-r"),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o"),
    max_steps: int | None = typer.Option(None, "--max-steps"),
    wandb: bool = typer.Option(False, "--wandb", help="Enable W&B logging."),
    run_name: str | None = typer.Option(None, "--run-name", "-r", help="Experiment name."),
    thinking: bool = typer.Option(False, "--thinking", help="Train with thinking mode."),
    grad_ckpt: bool = typer.Option(False, "--grad-ckpt", help="Enable gradient checkpointing."),
    qlora: bool = typer.Option(False, "--qlora", help="Load model in 4-bit (QLoRA)."),
):
    """Fine-tune the base model with LoRA (SFT)."""
    cfg = Config.from_yaml(
        config,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        lora_r=lora_r,
        output_dir=output_dir,
        max_steps=max_steps,
    )
    if wandb:
        cfg.wandb.enabled = True
    if run_name:
        cfg.wandb.run_name = run_name
    if thinking:
        cfg.model.enable_thinking = True
    if grad_ckpt:
        cfg.training.gradient_checkpointing = True
    if qlora:
        cfg.training.use_qlora = True

    t = cfg.training
    print(f"Training | epochs={t.num_epochs} lr={t.learning_rate} lora_r={cfg.lora.r}")

    from .train import main as train_main
    train_main(cfg)


@app.command()
def grpo(
    config: Path | None = _config_option,
    lora_r: int | None = typer.Option(None, "--lora-r"),
    wandb: bool = typer.Option(False, "--wandb", help="Enable W&B logging."),
    run_name: str | None = typer.Option(None, "--run-name", "-r", help="Experiment name."),
    thinking: bool = typer.Option(False, "--thinking", help="Train with thinking mode."),
):
    """Fine-tune with GRPO (reward-based optimization)."""
    cfg = Config.from_yaml(config, lora_r=lora_r)
    if wandb:
        cfg.wandb.enabled = True
    if run_name:
        cfg.wandb.run_name = run_name
    if thinking:
        cfg.model.enable_thinking = True

    g = cfg.grpo
    print(f"GRPO | epochs={g.num_epochs} lr={g.learning_rate} generations={g.num_generations}")

    from .train_grpo import main as grpo_main
    grpo_main(cfg)


@app.command()
def predict(
    query: str = typer.Argument(..., help="Natural-language filter query."),
    schema: Path = typer.Argument(..., help="Path to a JSON schema file."),
    config: Path | None = _config_option,
    temperature: float | None = typer.Option(None, "--temperature", "-t"),
    quantization: str = typer.Option("fp16", "--quantization", "-q"),
    thinking: bool = typer.Option(False, "--thinking"),
):
    """Generate a filter expression from a natural-language query."""
    from .inference import QUANTIZATION_MODES

    if quantization not in QUANTIZATION_MODES:
        print(f"Error: invalid quantization '{quantization}'. Choose from: {', '.join(QUANTIZATION_MODES)}")
        raise typer.Exit(1)
    if not schema.exists():
        print(f"Error: schema file not found: {schema}")
        raise typer.Exit(1)

    cfg = Config.from_yaml(config, temperature=temperature)
    if thinking:
        cfg.model.enable_thinking = True

    from .inference import load_model, predict as run_predict
    model, tokenizer = load_model(cfg, quantization=quantization)
    result = run_predict(query, str(schema), model=model, tokenizer=tokenizer, cfg=cfg)
    print(f"Query:   {query}")
    print(f"Filters: {result}")


@app.command()
def evaluate(
    config: Path | None = _config_option,
    max_samples: int | None = typer.Option(None, "--max-samples", "-n"),
    zero_shot: bool = typer.Option(False, "--zero-shot"),
    quantization: list[str] = typer.Option(["fp16"], "--quantization", "-q"),
    thinking: bool = typer.Option(False, "--thinking"),
):
    """Evaluate the model on held-out schemas."""
    from .inference import QUANTIZATION_MODES

    for q in quantization:
        if q not in QUANTIZATION_MODES:
            print(f"Error: invalid quantization '{q}'. Choose from: {', '.join(QUANTIZATION_MODES)}")
            raise typer.Exit(1)

    cfg = Config.from_yaml(config)
    if thinking:
        cfg.model.enable_thinking = True
    if zero_shot:
        print("Zero-shot evaluation (base model, no adapter)")

    from .evaluate import main as eval_main
    eval_main(cfg, max_samples=max_samples, zero_shot=zero_shot, quantizations=quantization)


@app.command()
def schemas(
    config: Path | None = _config_option,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """List available dataset schemas."""
    cfg = Config.from_yaml(config)
    if not cfg.paths.schema_dir.exists():
        print(f"Error: schema directory not found: {cfg.paths.schema_dir}")
        raise typer.Exit(1)

    schema_files = sorted(cfg.paths.schema_dir.glob("*.json"))
    if not schema_files:
        print("No schemas found.")
        raise typer.Exit(0)

    for sf in schema_files:
        with open(sf) as f:
            s = json.load(f)
        cols = s.get("columns", {})
        line = f"{s.get('name', sf.stem):<35s} {len(cols):>3d} cols  {s.get('row_count', '?'):>8} rows"
        if verbose:
            names = ", ".join(list(cols.keys())[:8])
            if len(cols) > 8:
                names += ", ..."
            line += f"  [{names}]"
        print(line)


@app.command(name="data-stats")
def data_stats(config: Path | None = _config_option):
    """Show training/eval dataset statistics."""
    from .data_loader import load_datasets

    cfg = Config.from_yaml(config)
    train_ds, eval_ds = load_datasets(cfg)

    print(f"Train:  {len(train_ds)}")
    print(f"Eval:   {len(eval_ds)}")
    print(f"Total:  {len(train_ds) + len(eval_ds)}")
    print(f"\nEval schemas: {', '.join(cfg.eval.schemas)}")
    if cfg.eval.exclude_schemas:
        print(f"Excluded:     {', '.join(cfg.eval.exclude_schemas)}")


@app.command(name="check-schemas")
def check_schemas(
    config: Path | None = _config_option,
    threshold: int = typer.Option(1024, "--threshold", "-t", help="Token length threshold."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show all schemas sorted by token count."),
):
    """Report schemas whose prompts exceed the token threshold."""
    from transformers import AutoTokenizer
    from .data_loader import format_schema, SYSTEM_PROMPT

    cfg = Config.from_yaml(config)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)

    results = []
    for schema_path in sorted(cfg.paths.schema_dir.glob("*.json")):
        with open(schema_path) as f:
            schema = json.load(f)
        name = schema["name"]
        schema_text = format_schema(schema)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"User query: show all\n\nSchema:\n{schema_text}"},
        ]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        n = len(tokenizer.encode(text))
        results.append((name, n))

    results.sort(key=lambda x: -x[1])
    over = [(n, t) for n, t in results if t > threshold]
    under = [(n, t) for n, t in results if t <= threshold]

    print(f"Threshold: {threshold} tokens\n")

    if verbose:
        print("All schemas (sorted by token count):")
        for name, n in results:
            flag = " <<<" if n > threshold else ""
            print(f"  {name:<45s} {n:>5d} tokens{flag}")
        print()

    if over:
        print(f"OVER threshold ({len(over)}):")
        for name, n in over:
            print(f"  {name:<45s} {n:>5d} tokens")
        print(f"\nAdd to config.yaml under eval.exclude_schemas:")
        print("  exclude_schemas:")
        for name, _ in over:
            print(f"    - {name}")
    else:
        print("All schemas are within the threshold.")

    print(f"\nOver: {len(over)}  |  Under: {len(under)}  |  Total: {len(results)}")
