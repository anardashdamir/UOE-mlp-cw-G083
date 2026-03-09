"""Load fine-tuned model and generate filter expressions."""

import json
import os

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import Config, QuantizationMode
from .data_loader import build_messages, format_schema

from typing import get_args

# Re-export for CLI validation
QUANTIZATION_MODES = get_args(QuantizationMode)


def _get_quant_config(quantization: str):
    """Return (BitsAndBytesConfig or None, model_kwargs) for the given mode."""
    if quantization == "int8":
        return BitsAndBytesConfig(load_in_8bit=True), {}
    elif quantization == "int4":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16"), {}
    # fp16 / default
    return None, {"torch_dtype": "auto"}


def load_model(cfg: Config = None, zero_shot: bool = False, quantization: str = "fp16"):
    """Load model with optional quantization.

    Args:
        cfg: Config object.
        zero_shot: If True, load base model without LoRA adapter.
        quantization: One of "fp16", "int8", "int4".
    """
    cfg = cfg or Config()
    quant_config, extra_kwargs = _get_quant_config(quantization)

    if zero_shot:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, device_map="auto", trust_remote_code=True,
            quantization_config=quant_config, **extra_kwargs,
        )
    else:
        adapter_path = str(cfg.adapter_dir)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, device_map="auto", trust_remote_code=True,
            quantization_config=quant_config, **extra_kwargs,
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        # Cannot merge_and_unload with quantized models
        if quantization == "fp16":
            model = model.merge_and_unload()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def predict(query: str, schema_path: str, model=None, tokenizer=None, cfg: Config = None):
    """Generate filter expression for a query given a schema file."""
    cfg = cfg or Config()

    if model is None or tokenizer is None:
        model, tokenizer = load_model(cfg)

    with open(schema_path) as f:
        schema = json.load(f)
    schema_text = format_schema(schema)

    messages = build_messages(query, schema_text)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen = cfg.generation
    output_ids = model.generate(
        **inputs,
        max_new_tokens=gen.max_new_tokens,
        temperature=gen.temperature,
        do_sample=gen.temperature > 0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.inference <query> <schema_path>")
        sys.exit(1)

    result = predict(sys.argv[1], sys.argv[2])
    print(f"Query:   {sys.argv[1]}")
    print(f"Filters: {result}")
