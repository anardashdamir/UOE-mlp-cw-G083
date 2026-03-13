"""Load data.json + schemas, format into chat messages, build HF Datasets."""

import json
import random

from datasets import Dataset

from .config import Config

SYSTEM_PROMPT = """You are a filter extraction engine. Given a natural language query and a dataset schema, output ONLY the corresponding structured filter expression.

SYNTAX RULES:
- Equality: column == 'value' (strings) or column == number (numeric/bool)
- Comparison: column > number, column >= number, column < number, column <= number
- Boolean: column == true, column == false
- Array contains: column CONTAINS 'value'
- Array contains all: column CONTAINS_ALL ['value1', 'value2']
- Combine filters with AND
- Use OR for alternatives: (column == 'a' OR column == 'b')
- Use parentheses to group OR clauses: (brand == 'toyota' OR brand == 'honda') AND price < 20000

RULES:
- ONLY use columns that exist in the provided schema
- Categorical and array values MUST be lowercase and match the schema exactly
- Do NOT invent columns or values not in the schema
- If the query mentions something not in the schema, ignore it
- Output ONLY the filter expression, nothing else"""


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


def build_messages(query: str, schema_text: str, filters: str = None, reasoning: str = None) -> list[dict]:
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

    for sample in raw_data:
        file_path = sample.get("file_path", "")
        if file_path.split("__")[-1] not in ("v0", "v1"):
            continue

        dataset_name = file_path.split("__")[0]
        if dataset_name in exclude_set:
            continue
        schema_text = schemas.get(dataset_name)
        if schema_text is None:
            continue

        reasoning = sample.get("reasoning") if cfg.model.enable_thinking else None
        messages = build_messages(sample["query"], schema_text, sample["filters"], reasoning)
        row = {"messages": messages}

        if dataset_name in eval_set:
            eval_rows.append(row)
        else:
            train_rows.append(row)

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
        if file_path.split("__")[-1] not in ("v0", "v1"):
            continue

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
