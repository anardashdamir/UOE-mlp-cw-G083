"""Load fine-tuned model and generate filter expressions."""

import json
from typing import get_args

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import Config, QuantizationMode
from .data_loader import build_messages, format_schema

QUANTIZATION_MODES = get_args(QuantizationMode)


def _get_quant_config(quantization: str):
    if quantization == "int8":
        return BitsAndBytesConfig(load_in_8bit=True), {}
    elif quantization == "int4":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16"), {}
    return None, {"dtype": "auto"}


def load_model(cfg: Config = None, zero_shot: bool = False, quantization: str = "fp16"):
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
        if quantization == "fp16":
            model = model.merge_and_unload()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
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
    output_ids = model.generate(
        **inputs,
        max_new_tokens=gen.max_new_tokens,
        temperature=gen.temperature,
        do_sample=gen.temperature > 0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
