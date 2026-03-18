"""Load fine-tuned model and generate filter expressions."""

import json
from pathlib import Path
from typing import get_args

from peft import PeftModel
from transformers import AutoTokenizer, BitsAndBytesConfig
from unsloth import FastLanguageModel

from .config import Config, QuantizationMode
from .data_loader import build_messages, format_schema
from .training_utils import strip_thinking_output

QUANTIZATION_MODES = get_args(QuantizationMode)


def load_model(
    cfg: Config = None,
    zero_shot: bool = False,
    quantization: str = "fp16",
    sft_adapter: str | None = None,
    grpo_adapter: str | None = None,
):
    """Load model with optional SFT and GRPO adapters.

    Args:
        sft_adapter: Path to SFT LoRA adapter directory.
        grpo_adapter: Path to GRPO LoRA adapter directory (requires sft_adapter).
    """
    cfg = cfg or Config()

    model, tokenizer = FastLanguageModel.from_pretrained(
        cfg.model.name,
        max_seq_length=cfg.training.max_seq_length,
        load_in_4bit=(quantization == "int4"),
        dtype=None,
    )

    if not zero_shot:
        sft_path = Path(sft_adapter) if sft_adapter else cfg.adapter_dir
        grpo_path = Path(grpo_adapter) if grpo_adapter else None

        if not sft_path.exists():
            raise FileNotFoundError(f"SFT adapter not found: {sft_path}")

        # Load and merge SFT adapter
        print(f"Loading SFT adapter from {sft_path}")
        model = PeftModel.from_pretrained(model, str(sft_path))
        model = model.merge_and_unload()

        # Load GRPO adapter on top
        if grpo_path:
            if not grpo_path.exists():
                raise FileNotFoundError(f"GRPO adapter not found: {grpo_path}")
            print(f"Loading GRPO adapter from {grpo_path}")
            model = PeftModel.from_pretrained(model, str(grpo_path))
            model = model.merge_and_unload()

    # Load tokenizer from adapter if available (has chat template)
    if not zero_shot:
        sft_path = Path(sft_adapter) if sft_adapter else cfg.adapter_dir
        if sft_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(sft_path), trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def predict(query: str, schema_path: str, model=None, tokenizer=None, cfg: Config = None):
    cfg = cfg or Config()
    if model is None or tokenizer is None:
        model, tokenizer = load_model(cfg)

    with open(schema_path) as f:
        schema = json.load(f)

    messages = build_messages(query, format_schema(schema))
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=cfg.model.enable_thinking,
        tokenize=True, return_dict=True, return_tensors="pt",
    ).to(model.device)

    gen = cfg.generation
    do_sample = gen.temperature > 0
    output_ids = model.generate(
        **inputs,
        max_new_tokens=gen.max_new_tokens,
        temperature=gen.temperature if do_sample else None,
        top_p=gen.top_p if do_sample else None,
        top_k=gen.top_k if do_sample else None,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if cfg.model.enable_thinking:
        result = strip_thinking_output(result)
    return result
