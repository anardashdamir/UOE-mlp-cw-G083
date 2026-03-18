"""Load data.json + schemas, format into chat messages, build HF Datasets."""

import json
import logging
import random

from datasets import Dataset

from .config import Config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You convert natural language queries into structured filter expressions based on a dataset schema.

OPERATORS:
  ==   Exact match         column == 'value'  |  column == 42  |  column == true
  !=   Not equal           column != 'value'
  > < >= <=                column > 100
  IN   Any of (same field) column IN ['a', 'b', 'c']
  NOT IN  Exclude set      column NOT IN ['x', 'y']

RULES:
1. Same field, multiple values → IN:  brand IN ['Nike', 'Adidas']
   NEVER use: brand == 'Nike' OR brand == 'Adidas'
2. OR only for different fields or different operators on the same field:
   (price < 50 OR rating > 4)  |  (num_pages > 400 OR num_pages < 150)
3. Parentheses required when mixing AND and OR: (A OR B) AND C
4. Strings in single quotes: col == 'value'. Numbers/booleans unquoted: col > 5, col == true
5. Values must exactly match the schema (case-sensitive).
6. Only filter on what the query explicitly asks — never add extra conditions.
7. If a query term has no matching schema field, ignore it.
8. If NO query terms match any schema field, output: EMPTY

NUMERIC THRESHOLDS:
- Explicit number in query → use that exact number: "under 50" → price < 50
- Vague word, no number → use the field's average from the schema:
  cheap/affordable/low/small/few/short/young → column < average
  expensive/premium/high/large/many/tall/old/senior → column > average
  popular/top/best/highly-rated/recent/modern/new → column > average
  worst/lowest/classic/vintage → column < average
- Explicit number always overrides vague words: "cheap, under 50" → price < 50

Output ONLY the filter expression."""


def format_schema(schema: dict) -> str:
    lines = []
    for col_name, col_info in schema["columns"].items():
        col_type = col_info["type"]
        if col_type == "categorical":
            values = ", ".join(col_info["values"])
            lines.append(f"{col_name}: categorical [{values}]")
        elif col_type in ("int", "float"):
            if "average" in col_info:
                lines.append(f"{col_name}: {col_type} (average={col_info['average']})")
            else:
                lines.append(f"{col_name}: {col_type}")
        elif col_type == "bool":
            lines.append(f"{col_name}: bool")
        elif col_type == "array":
            values = ", ".join(col_info.get("values", []))
            lines.append(f"{col_name}: array [{values}]")
        elif col_type == "str":
            lines.append(f"{col_name}: str")
        else:
            lines.append(f"{col_name}: {col_type}")
    return "\n".join(lines)


def build_messages(query: str, schema_text: str, filters: str = None, reasoning: str | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"User query: {query}\n\nSchema:\n{schema_text}"},
    ]
    if filters is not None:
        assistant_msg = {"role": "assistant", "content": filters}
        if reasoning:
            assistant_msg["reasoning_content"] = reasoning
        messages.append(assistant_msg)
    return messages


def _load_schemas(cfg: Config) -> dict[str, str]:
    schemas = {}
    for schema_path in cfg.paths.schema_dir.glob("*.json"):
        with open(schema_path) as f:
            schema = json.load(f)
        schemas[schema["name"]] = format_schema(schema)
    return schemas


def _load_split(cfg: Config, data_path, schemas: dict) -> list[dict]:
    """Load a single data file and convert to chat message rows."""
    with open(data_path) as f:
        raw_data = json.load(f)

    rows = []
    missing_schemas = set()

    for sample in raw_data:
        file_path = sample.get("file_path", "")
        dataset_name = file_path.split("__")[0]
        schema_text = schemas.get(dataset_name)
        if schema_text is None:
            missing_schemas.add(dataset_name)
            continue

        filters = sample["filters"].strip()
        if not filters:
            filters = "EMPTY"

        reasoning = sample.get("reasoning") if cfg.model.enable_thinking else None
        messages = build_messages(sample["query"], schema_text, filters, reasoning)
        rows.append({"messages": messages, "file_path": file_path})

    if missing_schemas:
        logger.warning(
            "Schemas referenced in data but missing from %s: %s (samples dropped)",
            cfg.paths.schema_dir, sorted(missing_schemas),
        )
    return rows


def load_datasets(cfg: Config = None) -> tuple[Dataset, Dataset]:
    """Load pre-split train and test datasets."""
    cfg = cfg or Config()
    schemas = _load_schemas(cfg)

    train_rows = _load_split(cfg, cfg.paths.train_path, schemas)
    eval_rows = _load_split(cfg, cfg.paths.test_path, schemas)

    random.seed(42)
    random.shuffle(train_rows)

    return Dataset.from_list(train_rows), Dataset.from_list(eval_rows)


def load_grpo_dataset(cfg: Config = None) -> Dataset:
    """Load prompts (without answers) for GRPO training from train split."""
    cfg = cfg or Config()
    schemas = _load_schemas(cfg)

    with open(cfg.paths.train_path) as f:
        raw_data = json.load(f)

    rows = []
    seen = set()

    for sample in raw_data:
        file_path = sample.get("file_path", "")
        dataset_name = file_path.split("__")[0]
        schema_text = schemas.get(dataset_name)
        if schema_text is None:
            continue

        key = (sample["query"], dataset_name)
        if key in seen:
            continue
        seen.add(key)

        prompt = build_messages(sample["query"], schema_text)
        rows.append({
            "prompt": prompt,
            "expected": sample["filters"],
            "schema_text": schema_text,
        })

    random.seed(42)
    random.shuffle(rows)
    return Dataset.from_list(rows)


if __name__ == "__main__":
    train_ds, eval_ds = load_datasets()
    print(f"Train: {len(train_ds)}, Test: {len(eval_ds)}")
    print("\n--- Sample ---")
    for msg in train_ds[0]["messages"]:
        print(f"[{msg['role']}]: {msg['content'][:200]}")
