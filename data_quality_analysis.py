"""Comprehensive data quality analysis for AutoFilter training data."""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


DATA_PATH = Path("data/data.json")
SCHEMA_DIR = Path("schemas")


def load_data():
    with open(DATA_PATH) as f:
        data = json.load(f)
    schemas = {}
    for p in SCHEMA_DIR.glob("*.json"):
        with open(p) as f:
            s = json.load(f)
        schemas[s["name"]] = s
    return data, schemas


def parse_file_path(fp):
    """Parse file_path like 'ab_test_results__easy__0__v3'."""
    parts = fp.rsplit("__", 3)
    if len(parts) == 4:
        schema, difficulty, idx, variant = parts
        return schema, difficulty, int(idx), variant
    return fp, "unknown", 0, "v0"


def analyze_basic_stats(data, schemas):
    print("=" * 70)
    print("1. BASIC STATISTICS")
    print("=" * 70)
    print(f"  Total samples:            {len(data):,}")
    print(f"  Total schemas in data:    {len(set(parse_file_path(d['file_path'])[0] for d in data))}")
    print(f"  Total schema files:       {len(schemas)}")

    difficulties = Counter(parse_file_path(d["file_path"])[1] for d in data)
    print(f"  Difficulty categories:    {len(difficulties)}")

    # Variants per base
    bases = defaultdict(set)
    for d in data:
        schema, diff, idx, var = parse_file_path(d["file_path"])
        bases[(schema, diff, idx)].add(var)
    var_counts = [len(v) for v in bases.values()]
    print(f"  Unique base prompts:      {len(bases):,}")
    print(f"  Variants per base:        avg={sum(var_counts)/len(var_counts):.1f}, min={min(var_counts)}, max={max(var_counts)}")
    print()


def analyze_difficulty_distribution(data):
    print("=" * 70)
    print("2. DIFFICULTY DISTRIBUTION")
    print("=" * 70)
    difficulties = Counter(parse_file_path(d["file_path"])[1] for d in data)
    total = len(data)
    for diff, count in sorted(difficulties.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {diff:40s} {count:5d} ({pct:5.1f}%) {bar}")
    print()


def analyze_schema_distribution(data, schemas):
    print("=" * 70)
    print("3. SCHEMA DISTRIBUTION")
    print("=" * 70)
    schema_counts = Counter(parse_file_path(d["file_path"])[0] for d in data)
    counts = sorted(schema_counts.values())
    print(f"  Schemas with data:        {len(schema_counts)}")
    print(f"  Samples per schema:       avg={sum(counts)/len(counts):.0f}, min={min(counts)}, max={max(counts)}, median={counts[len(counts)//2]}")

    # Schemas in data but no schema file
    data_schemas = set(schema_counts.keys())
    file_schemas = set(schemas.keys())
    missing_schemas = data_schemas - file_schemas
    unused_schemas = file_schemas - data_schemas
    if missing_schemas:
        print(f"\n  MISSING schema files ({len(missing_schemas)}):")
        for s in sorted(missing_schemas):
            print(f"    - {s} ({schema_counts[s]} samples)")
    if unused_schemas:
        print(f"\n  UNUSED schema files ({len(unused_schemas)}):")
        for s in sorted(unused_schemas):
            print(f"    - {s}")

    # Top and bottom schemas
    print(f"\n  Top 5 schemas:")
    for schema, count in schema_counts.most_common(5):
        print(f"    {schema:40s} {count:5d}")
    print(f"\n  Bottom 5 schemas:")
    for schema, count in schema_counts.most_common()[-5:]:
        print(f"    {schema:40s} {count:5d}")
    print()


def analyze_query_quality(data):
    print("=" * 70)
    print("4. QUERY QUALITY")
    print("=" * 70)
    queries = [d["query"] for d in data]
    lengths = [len(q) for q in queries]
    word_counts = [len(q.split()) for q in queries]

    print(f"  Query length (chars):     avg={sum(lengths)/len(lengths):.0f}, min={min(lengths)}, max={max(lengths)}")
    print(f"  Query length (words):     avg={sum(word_counts)/len(word_counts):.1f}, min={min(word_counts)}, max={max(word_counts)}")

    # Exact duplicates
    query_counter = Counter(queries)
    exact_dupes = sum(1 for c in query_counter.values() if c > 1)
    total_duped = sum(c for c in query_counter.values() if c > 1)
    print(f"\n  Exact duplicate queries:  {exact_dupes:,} unique queries appearing {total_duped:,} times total")
    if exact_dupes > 0:
        print(f"  Top duplicate queries:")
        for q, c in query_counter.most_common(5):
            if c > 1:
                print(f"    [{c}x] \"{q[:100]}{'...' if len(q) > 100 else ''}\"")

    # Empty or very short queries
    short = [q for q in queries if len(q.split()) < 4]
    if short:
        print(f"\n  Very short queries (<4 words): {len(short)}")
        for q in short[:5]:
            print(f"    \"{q}\"")

    # Queries by difficulty
    print(f"\n  Avg query length by difficulty:")
    diff_lengths = defaultdict(list)
    for d in data:
        diff = parse_file_path(d["file_path"])[1]
        diff_lengths[diff].append(len(d["query"].split()))
    for diff in sorted(diff_lengths.keys()):
        avg = sum(diff_lengths[diff]) / len(diff_lengths[diff])
        print(f"    {diff:40s} {avg:5.1f} words")
    print()


def analyze_filter_quality(data, schemas):
    print("=" * 70)
    print("5. FILTER EXPRESSION QUALITY")
    print("=" * 70)
    filters = [d["filters"] for d in data]

    # Operator usage
    ops = {
        "==": r"==",
        "!=": r"!=",
        ">": r"(?<!=)>(?!=)",
        ">=": r">=",
        "<": r"(?<!=)<(?!=)",
        "<=": r"<=",
        "AND": r"\bAND\b",
        "OR": r"\bOR\b",
        "CONTAINS": r"\bCONTAINS\b(?!_ALL)",
        "CONTAINS_ALL": r"\bCONTAINS_ALL\b",
        "IN": r"\bIN\b",
    }
    print("  Operator frequency:")
    for op_name, pattern in ops.items():
        count = sum(1 for f in filters if re.search(pattern, f))
        pct = count / len(filters) * 100
        print(f"    {op_name:20s} {count:6d} samples ({pct:5.1f}%)")

    # Filter complexity (number of conditions)
    and_counts = [len(re.findall(r"\bAND\b", f)) + 1 for f in filters]
    or_counts = [len(re.findall(r"\bOR\b", f)) for f in filters]
    print(f"\n  Conditions per filter (AND-separated):")
    print(f"    avg={sum(and_counts)/len(and_counts):.1f}, min={min(and_counts)}, max={max(and_counts)}")
    cond_dist = Counter(and_counts)
    for n in sorted(cond_dist.keys())[:8]:
        pct = cond_dist[n] / len(filters) * 100
        print(f"    {n} conditions: {cond_dist[n]:5d} ({pct:5.1f}%)")

    # Empty filters
    empty = [d for d in data if not d["filters"].strip()]
    if empty:
        print(f"\n  EMPTY filters: {len(empty)}")
        for d in empty[:3]:
            print(f"    query: \"{d['query'][:80]}\"")

    # Filters with potential syntax issues
    issues = []
    for d in data:
        f = d["filters"]
        # Unbalanced parens
        if f.count("(") != f.count(")"):
            issues.append(("unbalanced_parens", d))
        # Unbalanced quotes
        single_q = f.count("'")
        if single_q % 2 != 0:
            issues.append(("unbalanced_quotes", d))

    if issues:
        issue_types = Counter(t for t, _ in issues)
        print(f"\n  SYNTAX ISSUES found: {len(issues)}")
        for issue_type, count in issue_types.items():
            print(f"    {issue_type}: {count}")
            examples = [d for t, d in issues if t == issue_type][:3]
            for d in examples:
                print(f"      \"{d['filters'][:100]}\"")
    else:
        print(f"\n  No syntax issues detected (balanced parens & quotes)")

    # Check for IN operator (not in official syntax)
    in_usage = [d for d in data if re.search(r"\bIN\b", d["filters"])]
    if in_usage:
        print(f"\n  NOTE: {len(in_usage)} samples use IN operator (not in official SYNTAX RULES)")
        for d in in_usage[:3]:
            print(f"    \"{d['filters'][:100]}\"")
    print()


def analyze_query_filter_consistency(data):
    print("=" * 70)
    print("6. QUERY-FILTER CONSISTENCY (variant analysis)")
    print("=" * 70)

    # Group by base (schema__diff__idx)
    bases = defaultdict(list)
    for d in data:
        schema, diff, idx, var = parse_file_path(d["file_path"])
        bases[(schema, diff, idx)].append(d)

    # All variants of the same base should have the SAME filter
    mismatched = 0
    mismatch_examples = []
    for key, samples in bases.items():
        filter_set = set(s["filters"] for s in samples)
        if len(filter_set) > 1:
            mismatched += 1
            if len(mismatch_examples) < 3:
                mismatch_examples.append((key, samples))

    total_bases = len(bases)
    print(f"  Total base prompts:       {total_bases:,}")
    print(f"  Bases with SAME filter:   {total_bases - mismatched:,} ({(total_bases - mismatched)/total_bases*100:.1f}%)")
    print(f"  Bases with DIFF filters:  {mismatched:,} ({mismatched/total_bases*100:.1f}%)")

    if mismatch_examples:
        print(f"\n  Mismatch examples:")
        for key, samples in mismatch_examples:
            schema, diff, idx = key
            print(f"\n    Base: {schema}__{diff}__{idx}")
            for s in samples[:3]:
                print(f"      [{parse_file_path(s['file_path'])[3]}] query: \"{s['query'][:80]}\"")
                print(f"           filter: \"{s['filters'][:80]}\"")

    # Check if variant queries are actually different from each other
    identical_queries_in_variants = 0
    for key, samples in bases.items():
        query_set = set(s["query"] for s in samples)
        if len(query_set) < len(samples):
            identical_queries_in_variants += 1

    print(f"\n  Bases with duplicate queries across variants: {identical_queries_in_variants}")
    print()


def analyze_selected_fields(data, schemas):
    print("=" * 70)
    print("7. SELECTED FIELDS ANALYSIS")
    print("=" * 70)
    data_with_fields = [d for d in data if "selected_fields" in d and d["selected_fields"]]
    missing_fields = len(data) - len(data_with_fields)
    if missing_fields:
        print(f"  Samples MISSING selected_fields: {missing_fields}")
    field_counts = [len(d["selected_fields"]) for d in data_with_fields]
    print(f"  Fields per sample:        avg={sum(field_counts)/len(field_counts):.1f}, min={min(field_counts)}, max={max(field_counts)}")

    # Check if selected_fields match what's in the filter
    mismatches = 0
    mismatch_examples = []
    for d in data:
        if "selected_fields" not in d:
            continue
        schema_name = parse_file_path(d["file_path"])[0]
        schema = schemas.get(schema_name)
        if not schema:
            continue
        schema_cols = set(schema["columns"].keys())
        for field in d["selected_fields"]:
            if field not in schema_cols:
                mismatches += 1
                if len(mismatch_examples) < 3:
                    mismatch_examples.append((d, field, schema_name))
                break

    print(f"  Samples with fields NOT in schema: {mismatches}")
    if mismatch_examples:
        for d, field, schema_name in mismatch_examples:
            print(f"    schema={schema_name}, bad_field=\"{field}\", filter=\"{d['filters'][:60]}\"")

    # Fields used in filter but not in selected_fields
    filter_field_mismatches = 0
    for d in data:
        if "selected_fields" not in d:
            continue
        schema_name = parse_file_path(d["file_path"])[0]
        schema = schemas.get(schema_name)
        if not schema:
            continue
        schema_cols = set(schema["columns"].keys())
        # Extract field names from filter (words before operators)
        filter_fields = set()
        for match in re.finditer(r"(\w+)\s*(?:==|!=|>=?|<=?|CONTAINS|IN)", d["filters"]):
            candidate = match.group(1)
            if candidate in schema_cols:
                filter_fields.add(candidate)

        selected = set(d["selected_fields"])
        extra_in_filter = filter_fields - selected
        if extra_in_filter:
            filter_field_mismatches += 1

    print(f"  Samples with filter fields NOT in selected_fields: {filter_field_mismatches}")
    print()


def analyze_attempts(data):
    print("=" * 70)
    print("8. DATA GENERATION QUALITY (attempts)")
    print("=" * 70)
    attempts = [d["attempts"] for d in data]
    attempt_dist = Counter(attempts)
    print(f"  Attempts distribution:")
    for a in sorted(attempt_dist.keys()):
        count = attempt_dist[a]
        pct = count / len(data) * 100
        print(f"    {a} attempt(s): {count:6d} ({pct:5.1f}%)")

    # High-attempt samples may indicate generation issues
    high = [d for d in data if d["attempts"] > 2]
    if high:
        print(f"\n  Samples needing >2 attempts: {len(high)}")
        diffs = Counter(parse_file_path(d["file_path"])[1] for d in high)
        print(f"  By difficulty:")
        for diff, c in diffs.most_common(5):
            print(f"    {diff}: {c}")
    print()


def analyze_token_usage(data):
    print("=" * 70)
    print("9. TOKEN USAGE PATTERNS")
    print("=" * 70)
    prompt_tokens = [d["usage"]["prompt_tokens"] for d in data]
    completion_tokens = [d["usage"]["completion_tokens"] for d in data]
    total_tokens = [d["usage"]["total_tokens"] for d in data]

    print(f"  Prompt tokens:      avg={sum(prompt_tokens)/len(prompt_tokens):.0f}, min={min(prompt_tokens)}, max={max(prompt_tokens)}")
    print(f"  Completion tokens:  avg={sum(completion_tokens)/len(completion_tokens):.0f}, min={min(completion_tokens)}, max={max(completion_tokens)}")
    print(f"  Total tokens:       avg={sum(total_tokens)/len(total_tokens):.0f}, min={min(total_tokens)}, max={max(total_tokens)}")
    print(f"  Total token cost:   {sum(total_tokens):,} tokens across all samples")
    print()


def analyze_label_leakage(data):
    print("=" * 70)
    print("10. LABEL LEAKAGE CHECK")
    print("=" * 70)
    leaky = 0
    examples = []
    for d in data:
        query_lower = d["query"].lower()
        filters = d["filters"]
        # Check if the exact filter expression appears in the query
        if filters.lower() in query_lower:
            leaky += 1
            if len(examples) < 3:
                examples.append(d)
        # Check if query contains operator syntax
        elif re.search(r"(?:==|!=|>=|<=|CONTAINS_ALL|CONTAINS)\s", d["query"]):
            leaky += 1
            if len(examples) < 3:
                examples.append(d)

    print(f"  Queries containing filter syntax: {leaky}")
    if examples:
        for d in examples:
            print(f"    query: \"{d['query'][:100]}\"")
            print(f"    filter: \"{d['filters'][:80]}\"")
    print()


def analyze_diversity(data):
    print("=" * 70)
    print("11. DIVERSITY ANALYSIS")
    print("=" * 70)

    # Query starting patterns
    starts = Counter()
    for d in data:
        words = d["query"].split()[:3]
        starts[" ".join(words)] += 1

    print(f"  Unique 3-word query starts: {len(starts)}")
    print(f"  Top 10 query starts:")
    for start, count in starts.most_common(10):
        print(f"    \"{start}\" — {count}")

    # Filter pattern diversity
    # Normalize: replace values with placeholders
    def normalize_filter(f):
        f = re.sub(r"'[^']*'", "'X'", f)
        f = re.sub(r"\b\d+\.?\d*\b", "N", f)
        return f

    patterns = Counter(normalize_filter(d["filters"]) for d in data)
    print(f"\n  Unique filter patterns (normalized): {len(patterns)}")
    print(f"  Top 10 filter patterns:")
    for pat, count in patterns.most_common(10):
        print(f"    [{count:4d}] {pat[:80]}")

    # Difficulty vs filter complexity
    print(f"\n  Avg conditions by difficulty:")
    diff_complexity = defaultdict(list)
    for d in data:
        diff = parse_file_path(d["file_path"])[1]
        conds = len(re.findall(r"\bAND\b", d["filters"])) + 1
        diff_complexity[diff].append(conds)
    for diff in sorted(diff_complexity.keys()):
        avg = sum(diff_complexity[diff]) / len(diff_complexity[diff])
        print(f"    {diff:40s} {avg:.2f}")
    print()


def analyze_cross_schema_consistency(data, schemas):
    print("=" * 70)
    print("12. CROSS-SCHEMA CONSISTENCY")
    print("=" * 70)

    # Check if filter values match schema categorical values
    value_mismatches = 0
    mismatch_details = []

    for d in data:
        schema_name = parse_file_path(d["file_path"])[0]
        schema = schemas.get(schema_name)
        if not schema:
            continue

        # Extract 'value' from column == 'value'
        for match in re.finditer(r"(\w+)\s*==\s*'([^']*)'", d["filters"]):
            col, val = match.group(1), match.group(2)
            if col in schema["columns"]:
                col_info = schema["columns"][col]
                if col_info["type"] == "categorical" and "values" in col_info:
                    if val.lower() not in [v.lower() for v in col_info["values"]]:
                        value_mismatches += 1
                        if len(mismatch_details) < 5:
                            mismatch_details.append((schema_name, col, val, col_info["values"][:5]))

    print(f"  Filter values not in schema categoricals: {value_mismatches}")
    if mismatch_details:
        for schema_name, col, val, valid in mismatch_details:
            print(f"    schema={schema_name}, col={col}, got=\"{val}\", valid={valid}")

    # Schema coverage per difficulty
    diff_schemas = defaultdict(set)
    for d in data:
        schema, diff, _, _ = parse_file_path(d["file_path"])
        diff_schemas[diff].add(schema)
    print(f"\n  Schema coverage per difficulty:")
    for diff in sorted(diff_schemas.keys()):
        print(f"    {diff:40s} {len(diff_schemas[diff]):3d} schemas")
    print()


def summary(data, schemas):
    print("=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    issues = []

    # Check for missing schemas
    data_schemas = set(parse_file_path(d["file_path"])[0] for d in data)
    missing = data_schemas - set(schemas.keys())
    if missing:
        issues.append(f"  - {len(missing)} schemas referenced in data but no .json file found")

    # Check duplicates
    query_counter = Counter(d["query"] for d in data)
    dupes = sum(1 for c in query_counter.values() if c > 1)
    if dupes > 0:
        issues.append(f"  - {dupes} duplicate queries exist")

    # Check IN operator
    in_count = sum(1 for d in data if re.search(r"\bIN\b", d["filters"]))
    if in_count:
        issues.append(f"  - {in_count} samples use IN operator (not in SYNTAX RULES)")

    # Check high attempts
    high = sum(1 for d in data if d["attempts"] > 2)
    if high:
        issues.append(f"  - {high} samples needed >2 generation attempts")

    if issues:
        print("  Potential issues:")
        for i in issues:
            print(i)
    else:
        print("  No major issues detected!")
    print()


if __name__ == "__main__":
    data, schemas = load_data()
    analyze_basic_stats(data, schemas)
    analyze_difficulty_distribution(data)
    analyze_schema_distribution(data, schemas)
    analyze_query_quality(data)
    analyze_filter_quality(data, schemas)
    analyze_query_filter_consistency(data)
    analyze_selected_fields(data, schemas)
    analyze_attempts(data)
    analyze_token_usage(data)
    analyze_label_leakage(data)
    analyze_diversity(data)
    analyze_cross_schema_consistency(data, schemas)
    summary(data, schemas)
