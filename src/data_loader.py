"""Load data.json + schemas, format into chat messages, build HF Datasets."""

import json
import logging
import random

from datasets import Dataset

from .config import Config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a filter extraction engine. Given a natural language query and a dataset schema, output ONLY the corresponding structured filter expression.

OPERATORS:
- Equality:        column == 'value'  |  column == number  |  column == true/false
- Inequality:      column != 'value'  |  column != number
- Comparison:      column > number, column >= number, column < number, column <= number
- Membership:      column IN ['a', 'b', 'c']
- Exclusion:       column NOT IN ['a', 'b', 'c']
- Array contains:  column CONTAINS 'value'
- Array excludes:  column NOT CONTAINS 'value'
- Array all:       column CONTAINS_ALL ['value1', 'value2']
- Array any:       column CONTAINS_ANY ['value1', 'value2']

LOGICAL CONNECTORS:
- AND to combine conditions, OR for alternatives
- Parentheses are required when AND and OR appear together: (col == 'a' OR col == 'b') AND other > 10

VALUE RULES:
- String values use single quotes, numbers and booleans do not
- Booleans are lowercase: true, false
- Values must match the schema exactly (case-sensitive)

NUMERIC THRESHOLDS:
- When the query uses vague terms for numeric fields, use the schema median as pivot:
  "cheap/low/small/few" → column < median
  "expensive/high/large/many" → column > median
  "very cheap/budget" → column < (min + (median - min) * 0.25)
  "very expensive/premium" → column > (median + (max - median) * 0.75)
  "moderate/average/decent/mid-range" → column >= (median * 0.8) AND column <= (median * 1.2)
  "best/top/top-rated/popular/recent/modern" → column > median
  "worst/lowest/old/classic/vintage" → column < median
  "not too expensive/not too cheap" → opposite direction from median
- When the query states an explicit number, use that number exactly

EMPTY FILTER:
- If no query terms map to schema columns, output exactly: EMPTY

Output ONLY the filter expression, no explanation."""


def format_schema(schema: dict) -> str:
    lines = []
    for col_name, col_info in schema["columns"].items():
        col_type = col_info["type"]
        if col_type == "categorical":
            values = ", ".join(col_info["values"])
            lines.append(f"{col_name}: categorical [{values}]")
        elif col_type in ("int", "float"):
            stats = f"min={col_info['min']}, max={col_info['max']}"
            if "median" in col_info:
                stats += f", median={col_info['median']}"
            lines.append(f"{col_name}: {col_type} ({stats})")
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


def load_datasets(cfg: Config = None) -> tuple[Dataset, Dataset]:
    """Split by schema name: eval schemas go to eval, rest to train."""
    cfg = cfg or Config()
    schemas = _load_schemas(cfg)

    with open(cfg.paths.data_path) as f:
        raw_data = json.load(f)

    eval_set = set(cfg.eval.schemas)
    exclude_set = set(cfg.eval.exclude_schemas)
    train_rows, eval_rows = [], []
    missing_schemas = set()

    for sample in raw_data:
        file_path = sample.get("file_path", "")
        dataset_name = file_path.split("__")[0]
        if dataset_name in exclude_set:
            continue
        schema_text = schemas.get(dataset_name)
        if schema_text is None:
            missing_schemas.add(dataset_name)
            continue

        # Normalize empty filters to the EMPTY sentinel
        filters = sample["filters"].strip()
        if not filters:
            filters = "EMPTY"

        reasoning = sample.get("reasoning") if cfg.model.enable_thinking else None
        messages = build_messages(sample["query"], schema_text, filters, reasoning)
        row = {"messages": messages, "file_path": file_path}

        if dataset_name in eval_set:
            eval_rows.append(row)
        else:
            train_rows.append(row)

    if missing_schemas:
        logger.warning(
            "Schemas referenced in data but missing from %s: %s (samples dropped)",
            cfg.paths.schema_dir, sorted(missing_schemas),
        )

    random.seed(42)
    random.shuffle(train_rows)

    return Dataset.from_list(train_rows), Dataset.from_list(eval_rows)


def load_grpo_dataset(cfg: Config = None) -> Dataset:
    """Load prompts (without answers) for GRPO training."""
    cfg = cfg or Config()
    schemas = _load_schemas(cfg)

    with open(cfg.paths.data_path) as f:
        raw_data = json.load(f)

    eval_set = set(cfg.eval.schemas)
    exclude_set = set(cfg.eval.exclude_schemas)
    rows = []
    seen = set()

    for sample in raw_data:
        file_path = sample.get("file_path", "")
        dataset_name = file_path.split("__")[0]
        if dataset_name in eval_set or dataset_name in exclude_set:
            continue
        schema_text = schemas.get(dataset_name)
        if schema_text is None:
            continue

        # Deduplicate: same query+schema = same prompt
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
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    print("\n--- Sample ---")
    for msg in train_ds[0]["messages"]:
        print(f"[{msg['role']}]: {msg['content'][:200]}")
