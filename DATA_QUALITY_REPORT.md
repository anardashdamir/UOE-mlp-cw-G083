# Data Quality Report

**Dataset:** `data/data.json`
**Generated:** 2026-03-11
**Script:** `data_quality_analysis.py`

---

## 1. Overview

| Metric | Value |
|---|---|
| Total samples | 28,820 |
| Unique schemas | 251 |
| Schema files on disk | 249 |
| Difficulty categories | 23 |
| Unique base prompts | 5,764 |
| Variants per base | 5 (constant) |
| Avg query length | 19.4 words |
| Avg filter conditions | 3.3 |

---

## 2. Difficulty Distribution

All 23 categories are nearly perfectly balanced (~4.3–4.4% each).

| Difficulty | Samples | % | Avg Words | Avg Conditions |
|---|---|---|---|---|
| easy | 1,250 | 4.3% | 9.8 | 1.54 |
| medium | 1,255 | 4.4% | 16.3 | 3.08 |
| hard | 1,250 | 4.3% | 28.5 | 5.49 |
| mix | 1,255 | 4.4% | 26.1 | 4.77 |
| or_easy | 1,250 | 4.3% | 15.8 | 2.39 |
| or_medium | 1,255 | 4.4% | 22.9 | 3.72 |
| or_hard | 1,255 | 4.4% | 35.6 | 5.82 |
| or_mixed_easy | 1,255 | 4.4% | 26.2 | 3.94 |
| or_mixed_medium | 1,255 | 4.4% | 33.1 | 5.69 |
| or_mixed_hard | 1,255 | 4.4% | 37.0 | 6.12 |
| array_easy | 1,255 | 4.4% | 11.2 | 1.96 |
| array_medium | 1,250 | 4.3% | 17.8 | 3.51 |
| array_hard | 1,255 | 4.4% | 24.9 | 4.93 |
| negative | 1,255 | 4.4% | 14.4 | 2.44 |
| cross_schema | 1,255 | 4.4% | 19.8 | 2.76 |
| hallucination | 1,255 | 4.4% | 18.5 | 2.60 |
| hallucination_full | 1,250 | 4.3% | 13.0 | 1.00 |
| adversarial | 1,250 | 4.3% | 12.9 | 2.36 |
| adversarial_typos | 1,255 | 4.4% | 10.9 | 2.01 |
| adversarial_slang | 1,250 | 4.3% | 14.2 | 2.00 |
| adversarial_abbreviations | 1,245 | 4.3% | 9.7 | 2.15 |
| adversarial_numbers_as_words | 1,255 | 4.4% | 13.3 | 2.11 |
| adversarial_mixed | 1,255 | 4.4% | 15.1 | 2.96 |

Complexity scales correctly with difficulty: easy (1.5 conditions) → hard (5.5) → or_mixed_hard (6.1).

---

## 3. Schema Distribution

- **251 schemas** represented in data, fairly uniform (100–115 samples each)
- **Median samples per schema:** 115
- **Min/Max:** 100 (`yoga_studios`) / 115 (most schemas)

### Missing Schema Files (3)

These schemas appear in data but have no corresponding `.json` file in `schemas/`. Their samples are **silently dropped** during training.

| Schema | Samples Lost |
|---|---|
| `board_games_catalog_v2` | 115 |
| `cosmetics_csv` | 115 |
| `f1_race_results_detailed` | 115 |
| **Total** | **345** |

### Unused Schema Files (1)

| Schema File | Note |
|---|---|
| `cosmetics.csv.json` | Likely naming mismatch with `cosmetics_csv` in data |

---

## 4. Query Quality

| Metric | Value |
|---|---|
| Avg length (chars) | 112 |
| Avg length (words) | 19.4 |
| Min / Max words | 1 / 90 |
| Exact duplicate queries | 1 (`"rating > 3.5"` appears 2x) |
| Very short queries (<4 words) | 313 |

### Short Query Examples

```
"rpv over 50"
"lounges in Singapore"
"BS only"
"anything b4 1996"
```

Short queries are concentrated in adversarial categories (abbreviations, slang) — this is by design.

### Top Query Starts (diversity check)

| Pattern | Count |
|---|---|
| "I'm looking for" | 1,265 |
| "Do you have" | 1,262 |
| "I need a" | 603 |
| "Can you find" | 597 |
| "Looking for a" | 388 |

16,043 unique 3-word starts across 28,820 samples — good diversity.

---

## 5. Filter Expression Quality

### Operator Usage

| Operator | Samples | % |
|---|---|---|
| `==` | 21,585 | 74.9% |
| `>` | 13,460 | 46.7% |
| `<` | 12,880 | 44.7% |
| `>=` | 8,545 | 29.6% |
| `AND` | 23,485 | 81.5% |
| `<=` | 2,970 | 10.3% |
| `OR` | 7,645 | 26.5% |
| `!=` | 970 | 3.4% |
| `CONTAINS` | 465 | 1.6% |
| `IN` | 3,955 | 13.7% |
| `CONTAINS_ALL` | 45 | 0.2% |

### Condition Complexity

| Conditions | Samples | % |
|---|---|---|
| 1 | 5,335 | 18.5% |
| 2 | 6,205 | 21.5% |
| 3 | 6,105 | 21.2% |
| 4 | 4,485 | 15.6% |
| 5 | 2,940 | 10.2% |
| 6 | 2,015 | 7.0% |
| 7 | 1,140 | 4.0% |
| 8+ | 595 | 2.1% |

### Unique Filter Patterns (normalized)

5,278 unique patterns after replacing values with placeholders — strong diversity.

---

## 6. Variant Consistency

| Check | Result |
|---|---|
| Bases with same filter across all 5 variants | **5,764 / 5,764 (100%)** |
| Bases with duplicate queries across variants | **0** |

All variants produce unique query rephrasings while preserving the same target filter.

---

## 7. Data Generation Quality

| Attempts | Samples | % |
|---|---|---|
| 1 | 28,395 | 98.5% |
| 2 | 390 | 1.4% |
| 3 | 35 | 0.1% |

98.5% generated on first attempt — very clean data generation pipeline.

---

## 8. Token Usage

| Metric | Avg | Min | Max |
|---|---|---|---|
| Prompt tokens | 1,692 | 1,353 | 4,068 |
| Completion tokens | 204 | 47 | 617 |
| Total tokens | 1,896 | 1,456 | 4,285 |

**Total token cost:** 54.6M tokens across all samples.

---

## 9. Issues Found

### HIGH — `IN` Operator Not in SYNTAX RULES

- **3,955 samples (13.7%)** use `IN` / `NOT IN` operators
- The system prompt SYNTAX RULES only define `==`, comparisons, `AND`, `OR`, `CONTAINS`, `CONTAINS_ALL`
- **Impact:** Model learns an operator that isn't documented in the prompt it receives at inference

```
cadr_rating > 200 AND filter_type NOT IN ['Carbon', 'UV'] AND has_app_control == true
```

**Fix options:**
1. Add `IN` / `NOT IN` to SYNTAX RULES in the system prompt
2. Convert `IN` samples to `OR` chains: `(col == 'a' OR col == 'b')`

---

### MEDIUM — Empty Filters in `hallucination_full`

- **1,265 samples** have empty filter strings (all from `hallucination_full` difficulty)
- These produce empty assistant responses during training
- **Likely intentional** — model should learn to output nothing for nonsensical queries
- **Verify:** confirm empty output is the desired behavior, or use a sentinel like `NO_FILTER`

---

### MEDIUM — Label Leakage in Queries

- **1,163 samples (4.0%)** contain filter syntax directly in the query text

Examples:
```
"listing price < 10000 & gender == 'Men' - gimme those"
"Filter for (user_segment == 'New' OR user_segment == 'Returning') with ..."
```

- Concentrated in adversarial categories where queries intentionally mix natural language with syntax
- **Impact:** Model may learn to copy syntax from input rather than truly understanding the query
- **Recommendation:** Review if this is desired adversarial training behavior or unintended leakage

---

### MEDIUM — 60 Filters with Unbalanced Quotes

All caused by escaped apostrophes in values like `Women\'s Accessories`:

```
category_name IN ['Women\'s Accessories', 'Women\'s Handbags']
```

- **Impact:** Filter parser may fail on these at evaluation time
- **Fix:** Use double quotes for values containing apostrophes, or escape differently

---

### MEDIUM — 3 Missing Schema Files

345 samples reference schemas with no `.json` file. These are silently dropped by `data_loader.py`.

| Missing Schema | Likely Match |
|---|---|
| `cosmetics_csv` | `cosmetics.csv.json` (naming mismatch) |
| `board_games_catalog_v2` | `board_games_catalog.json` (version mismatch) |
| `f1_race_results_detailed` | `f1_race_results.json` (variant mismatch) |

---

### LOW — 1,250 Samples Missing `selected_fields`

All from `hallucination_full` category. Since these have empty filters, missing `selected_fields` is consistent.

---

### LOW — 105 Samples with Fields Not in Schema

All from `f1_race_results` — field `fastest_lap_rank` referenced in `selected_fields` but not in the schema file.

---

### LOW — 715 Samples with Filter Fields Not in `selected_fields`

Metadata inconsistency — fields used in the filter expression aren't listed in `selected_fields`. Does not affect training (only `filters` and `query` are used).

---

## 10. Summary

| Aspect | Status |
|---|---|
| Sample count | 28,820 — sufficient |
| Balance (difficulty) | Excellent — near-uniform |
| Balance (schema) | Excellent — 100-115 per schema |
| Variant quality | Perfect — 100% filter consistency, 0 duplicate queries |
| Query diversity | Good — 16K unique starts |
| Filter diversity | Good — 5.3K unique patterns |
| Generation quality | Excellent — 98.5% first-attempt |
| Syntax correctness | 99.8% (60 quote issues) |
| Operator coverage | Good, but `IN` undocumented |
| Label leakage | 4.0% — review if intentional |

### Priority Fixes

1. **Add `IN`/`NOT IN` to SYNTAX RULES** or convert 3,955 samples to `OR` chains
2. **Fix 3 missing schema files** (rename or add) to recover 345 samples
3. **Fix 60 escaped-quote filters** (`Women\'s` → proper quoting)
4. **Decide on `hallucination_full` empty outputs** — keep or use sentinel
5. **Review 1,163 label-leaky queries** — confirm adversarial intent
