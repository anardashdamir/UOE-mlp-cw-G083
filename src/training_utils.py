"""Shared helpers for training and evaluation."""

import inspect
from datetime import datetime

from .config import Config


def build_run_name(cfg: Config, prefix: str = "") -> str:
    model_short = cfg.model.name.split("/")[-1].lower()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"{prefix}_" if prefix else ""
    if prefix == "grpo":
        lr = cfg.grpo.learning_rate
        ep = cfg.grpo.num_epochs
    else:
        lr = cfg.training.learning_rate
        ep = cfg.training.num_epochs
    return f"{tag}{model_short}_r{cfg.lora.r}_a{cfg.lora.alpha}_lr{lr:.0e}_ep{ep}_{ts}"


def disable_thinking(tokenizer):
    if "enable_thinking" not in inspect.signature(tokenizer.apply_chat_template).parameters:
        return
    original = tokenizer.apply_chat_template
    def wrapper(*args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return original(*args, **kwargs)
    tokenizer.apply_chat_template = wrapper


def enable_thinking(tokenizer):
    if "enable_thinking" not in inspect.signature(tokenizer.apply_chat_template).parameters:
        return
    original = tokenizer.apply_chat_template
    def wrapper(*args, **kwargs):
        kwargs.setdefault("enable_thinking", True)
        return original(*args, **kwargs)
    tokenizer.apply_chat_template = wrapper


def strip_thinking_output(text: str) -> str:
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if "\nassistant" in text:
        text = text.split("\nassistant")[-1].strip()
    if text.startswith("assistant"):
        text = text[len("assistant"):].strip()
    return text


def setup_logging(cfg: Config, run_name: str):
    if cfg.wandb.enabled:
        import wandb
        if cfg.wandb.wandb_api_key:
            wandb.login(key=cfg.wandb.wandb_api_key)
        wandb.init(
            project=cfg.wandb.wandb_project,
            name=run_name,
            config=cfg.model_dump(exclude={"paths", "wandb"}),
        )
        return "wandb", None
    return "tensorboard", str(cfg.paths.output_dir / "tb_logs" / run_name)
