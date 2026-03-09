"""Nested configuration loaded from config.yaml."""

from pathlib import Path

import yaml
from typing import Literal

from pydantic import BaseModel, Field


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config.yaml"


QuantizationMode = Literal["fp16", "int8", "int4"]


class ModelConfig(BaseModel):
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    quantization: QuantizationMode = "fp16"


class LoraConfig(BaseModel):
    r: int = Field(16, gt=0)
    alpha: int = Field(32, gt=0)
    dropout: float = Field(0.05, ge=0.0, le=1.0)
    target_modules: list[str] = ["q_proj", "v_proj"]


class WandbConfig(BaseModel):
    enabled: bool = True
    project: str = "autofilter"
    run_name: str | None = None
    api_key: str | None = None


class TrainingConfig(BaseModel):
    num_epochs: int = Field(3, gt=0)
    batch_size: int = Field(8, gt=0)
    gradient_accumulation_steps: int = Field(2, gt=0)
    learning_rate: float = Field(2e-4, gt=0)
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = Field(0.05, ge=0.0, le=1.0)
    max_seq_length: int = Field(512, gt=0)
    max_steps: int = -1


class GenerationConfig(BaseModel):
    max_new_tokens: int = Field(128, gt=0)
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    eval_batch_size: int = Field(16, gt=0)


class PathsConfig(BaseModel):
    base_dir: Path = _PROJECT_ROOT
    data_path: Path | None = None
    schema_dir: Path | None = None
    output_dir: Path | None = None


class EvalConfig(BaseModel):
    schemas: list[str] = [
        "hotel_bookings", "job_listings", "restaurant_business_rankings_2020",
        "wine_reviews", "zoo_animals", "diabetes", "diamonds",
        "electric_vehicles", "nba_players", "youtube_statistics",
    ]


class Config(BaseModel):
    model: ModelConfig = ModelConfig()
    lora: LoraConfig = LoraConfig()
    training: TrainingConfig = TrainingConfig()
    generation: GenerationConfig = GenerationConfig()
    paths: PathsConfig = PathsConfig()
    eval: EvalConfig = EvalConfig()
    wandb: WandbConfig = WandbConfig()

    def model_post_init(self, __context):
        p = self.paths
        if p.data_path is None:
            p.data_path = p.base_dir / "data" / "data.json"
        if p.schema_dir is None:
            p.schema_dir = p.base_dir / "schemas"
        if p.output_dir is None:
            p.output_dir = p.base_dir / "output"

    @property
    def adapter_dir(self) -> Path:
        return self.paths.output_dir / "final_adapter"

    @classmethod
    def from_yaml(cls, path: Path | str | None = None, **overrides) -> "Config":
        """Load from YAML, then apply flat CLI overrides."""
        path = Path(path) if path else _DEFAULT_CONFIG
        data: dict = {}
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}

        # Drop None sections (happens when all keys are commented out)
        data = {k: v for k, v in data.items() if v is not None}

        # Map flat CLI overrides into the nested structure
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
