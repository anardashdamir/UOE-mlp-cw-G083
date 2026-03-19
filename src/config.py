"""Nested configuration loaded from config.yaml."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config.yaml"

QuantizationMode = Literal["fp16", "int8", "int4"]


class ModelConfig(BaseModel):
    name: str = "Qwen/Qwen3.5-4B"
    quantization: QuantizationMode = "fp16"
    enable_thinking: bool = False


class LoraConfig(BaseModel):
    r: int = Field(16, gt=0)
    alpha: int = Field(32, gt=0)
    dropout: float = Field(0.05, ge=0.0, le=1.0)
    target_modules: str | list[str] = "all-linear"


class WandbConfig(BaseSettings):
    enabled: bool = False
    wandb_project: str = "autofilter"
    run_name: str | None = None
    wandb_api_key: str | None = None

    model_config = {"env_file": _PROJECT_ROOT / "wandb.env", "extra": "ignore"}


class TrainingConfig(BaseModel):
    num_epochs: int = Field(5, gt=0)
    batch_size: int = Field(2, gt=0)
    gradient_accumulation_steps: int = Field(16, gt=0)
    learning_rate: float = Field(1e-4, gt=0)
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = Field(100, ge=0)
    max_seq_length: int = Field(1536, gt=0)
    max_steps: int = -1
    gradient_checkpointing: bool = True
    use_qlora: bool = False
    logging_steps: int = Field(10, gt=0)


class GRPOConfig(BaseModel):
    num_epochs: int = Field(1, gt=0)
    batch_size: int = Field(4, gt=0)
    gradient_accumulation_steps: int = Field(4, gt=0)
    learning_rate: float = Field(5e-6, gt=0)
    max_completion_length: int = Field(256, gt=0)
    num_generations: int = Field(4, gt=0)
    beta: float = Field(0.1, ge=0.0)
    max_steps: int = Field(-1)


class GenerationConfig(BaseModel):
    max_new_tokens: int = Field(512, gt=0)
    temperature: float = Field(0.6, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(20, ge=0)
    eval_batch_size: int = Field(16, gt=0)


class PathsConfig(BaseModel):
    base_dir: Path = _PROJECT_ROOT
    train_path: Path | None = None
    test_path: Path | None = None
    schema_dir: Path | None = None
    output_dir: Path | None = None


class Config(BaseModel):
    model: ModelConfig = ModelConfig()
    lora: LoraConfig = LoraConfig()
    training: TrainingConfig = TrainingConfig()
    grpo: GRPOConfig = GRPOConfig()
    generation: GenerationConfig = GenerationConfig()
    paths: PathsConfig = PathsConfig()
    wandb: WandbConfig = WandbConfig()

    def model_post_init(self, __context):
        p = self.paths
        if p.train_path is None:
            p.train_path = p.base_dir / "data" / "train.json"
        if p.test_path is None:
            p.test_path = p.base_dir / "data" / "test.json"
        if p.schema_dir is None:
            p.schema_dir = p.base_dir / "schemas"
        if p.output_dir is None:
            p.output_dir = p.base_dir / "output"

    @property
    def adapter_dir(self) -> Path:
        model_short = self.model.name.split("/")[-1].lower()
        thinking = "thinking" if self.model.enable_thinking else "no_thinking"
        return self.paths.output_dir / f"{model_short}_{thinking}"

    @classmethod
    def from_yaml(cls, path: Path | str | None = None, **overrides) -> "Config":
        path = Path(path) if path else _DEFAULT_CONFIG
        data: dict = {}
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}

        data = {k: v for k, v in data.items() if v is not None}

        mapping = {
            "num_epochs": ("training", "num_epochs"),
            "batch_size": ("training", "batch_size"),
            "learning_rate": ("training", "learning_rate"),
            "max_steps": ("training", "max_steps"),
            "lora_r": ("lora", "r"),
            "temperature": ("generation", "temperature"),
            "output_dir": ("paths", "output_dir"),
        }
        for key, val in overrides.items():
            if val is None:
                continue
            if key in mapping:
                section, field = mapping[key]
                data.setdefault(section, {})[field] = val
            else:
                data[key] = val

        return cls.model_validate(data)
