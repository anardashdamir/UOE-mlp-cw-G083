"""Shared parsing utilities for filter expressions.

All metrics should import parsing helpers from here
to ensure consistent normalization across metrics.
"""

import re


def strip_outer_parens(s: str) -> str:
    """Remove outer parentheses if they wrap the entire expression."""
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        balanced = True
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0 and i < len(s) - 1:
                balanced = False
                break
        if balanced:
            s = s[1:-1].strip()
        else:
            break
    return s


def split_top_level_and(filter_str: str) -> list[str]:
    """Split on AND only at parenthesis depth 0 and outside quoted strings."""
    filter_str = strip_outer_parens(filter_str)
    parts = []
    depth = 0
    in_quotes = None
    current = []
    i = 0
    while i < len(filter_str):
        ch = filter_str[i]
        if ch in ('"', "'") and in_quotes is None:
            in_quotes = ch
            current.append(ch)
        elif ch == in_quotes:
            in_quotes = None
            current.append(ch)
        elif in_quotes is None and ch == "(":
            depth += 1
            current.append(ch)
        elif in_quotes is None and ch == ")":
            depth -= 1
            current.append(ch)
        elif in_quotes is None and depth == 0:
            m = re.match(r"\s+AND\s+", filter_str[i:], re.IGNORECASE)
            if m:
                parts.append("".join(current))
                current = []
                i += len(m.group(0))
                continue
            else:
                current.append(ch)
        else:
            current.append(ch)
        i += 1
    if current:
        parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def normalize_clause(clause: str) -> str:
    """Normalize a single clause: whitespace, parens, numeric values, OR ordering."""
    clause = clause.strip().lower()
    clause = strip_outer_parens(clause)
    # Collapse whitespace
    clause = re.sub(r"\s+", " ", clause)
    # Normalize spaces around operators
    clause = re.sub(r"\s*(==|!=|>=|<=|>|<)\s*", r" \1 ", clause)
    clause = re.sub(r"\s*(contains_all|contains)\s*", r" \1 ", clause, flags=re.IGNORECASE)
    # Normalize trailing .0 on numbers: 4.0 -> 4
    clause = re.sub(r"\b(\d+)\.0\b", r"\1", clause)
    # Sort OR operands for order-independent comparison
    if " or " in clause:
        operands = re.split(r"\s+or\s+", clause)
        operands = sorted(op.strip() for op in operands)
        clause = " or ".join(operands)
    return clause.strip()


def parse_filters(filter_str: str) -> set[str]:
    """Split filter string into normalized clauses (paren-aware, whitespace-normalized)."""
    clauses = split_top_level_and(filter_str.strip())
    return {normalize_clause(c) for c in clauses if c.strip()}


def extract_fields(filter_str: str) -> list[str]:
    """Extract field names from filter clauses (left side of operators)."""
    clauses = split_top_level_and(filter_str.strip())
    fields = []
    for clause in clauses:
        clause = strip_outer_parens(clause.strip().lower())
        if not clause:
            continue
        # For OR groups, extract fields from each operand
        if " or " in clause.lower():
            for operand in re.split(r"\s+or\s+", clause, flags=re.IGNORECASE):
                operand = strip_outer_parens(operand.strip())
                m = re.match(r"(.+?)\s*(?:==|!=|>=|<=|>|<|\bIN\b|\bCONTAINS\b)", operand, re.IGNORECASE)
                if m:
                    fields.append(m.group(1).strip().lower())
        else:
            m = re.match(r"(.+?)\s*(?:==|!=|>=|<=|>|<|\bIN\b|\bCONTAINS\b)", clause, re.IGNORECASE)
            if m:
                fields.append(m.group(1).strip().lower())
    return fields


def extract_schema_columns(user_content: str) -> set[str]:
    """Extract column names from the schema text in the user message."""
    columns = set()
    in_schema = False
    for line in user_content.split("\n"):
        if "<schema>" in line:
            in_schema = True
            continue
        if "</schema>" in line:
            break
        if in_schema and ":" in line:
            col_name = line.split(":")[0].strip().lower()
            if col_name:
                columns.add(col_name)
    return columns


def count_clauses(filter_str: str) -> int:
    """Count the number of top-level AND clauses in a filter expression."""
    return len(split_top_level_and(filter_str.strip()))


def is_valid_syntax(filter_str: str) -> bool:
    """Check if a filter expression has basic valid syntax."""
    filter_str = filter_str.strip()
    if not filter_str:
        return False
    # Check balanced parentheses
    depth = 0
    for ch in filter_str:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth < 0:
            return False
    if depth != 0:
        return False
    # Check at least one operator exists
    if not re.search(r"(==|!=|>=|<=|>|<|\bCONTAINS\b|\bCONTAINS_ALL\b|\bIN\b)", filter_str, re.IGNORECASE):
        return False
    return True
