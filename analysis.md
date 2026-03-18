# Training Data Quality Analysis — Full Dataset

**Date**: 2026-03-16
**Dataset**: data.json (3,291 samples)
**Analysis**: Every single sample checked programmatically

---

## Summary

| Issue Type | Count | Samples Affected | Severity |
|-----------|-------|------------------|----------|
| OR instead of IN (same field) | 551 | 551 (16.7%) | Medium |
| Tautological OR (matches all values) | 59 | 59 (1.8%) | High |
| Direction contradiction (older→age<median) | 35 | 29 (0.9%) | **Critical** |
| Missing parentheses (AND/OR ambiguity) | 16 | 16 (0.5%) | High |
| Wrong operator for column type | 12 | 12 (0.4%) | High |
| False EMPTY (valid query → EMPTY) | 11 | 11 (0.3%) | **Critical** |
| Impossible filter (exceeds min/max) | 11 | 11 (0.3%) | High |
| Redundant OR (x>5 OR x>10) | 5 | 5 (0.2%) | Medium |
| Date as integer | 1 | 1 (0.0%) | Low |
| **Total** | **701** | **629 (19.1%)** | |

**19.1% of training data has at least one issue.**

---

## Issue 1: Direction Contradictions (35 instances, CRITICAL)

Query says "older/elderly/senior" but filter uses `age < median`. The model learns that "older" can mean EITHER direction.

**All instances:**

| # | Sample | Query word | Filter | Median |
|---|--------|-----------|--------|--------|
| 17 | bank_transactions__or_easy__0 | "under 30" (older context) | customer_age < 30 | 45 |
| 247 | gym_members__or_easy__0 | "over 50" (older context) | age < 25 | 40 |
| 265 | healthcare__adversarial_mixed__0 | "older" | age < 39 | 39 |
| 348 | healthcare__or_mixed_hard__0 | "older" | age < 39 | 39 |
| 395 | gym_members__adversarial_mixed__0 | "older" | age < 35 | 40 |
| 585 | gym_members__or_medium__1 | "older" | age < 25 | 40 |
| 673 | healthcare__or_mixed_easy__1 | "older" | age < 25 | 39 |
| 774 | gym_members__or_mixed_medium__1 | "older" | age < 30 | 40 |
| 996 | bank_transactions__adversarial__2 | "older" | customer_age < 45 | 45 |
| 1116 | gym_members__cross_schema__2 | "older" | age < 40 | 40 |
| 1119 | gym_members__or_mixed_medium__2 | "older" | age < 30 | 40 |
| 1120 | healthcare__or_mixed_hard__2 | "older" | age < 30 | 39 |
| 1197 | gym_members__mix__2 | "older" | age < 40 | 40 |
| 1321 | gym_members__or_mixed_easy__2 | "older" | age < 25 | 40 |
| 1892 | healthcare__or_mixed_easy__3 | "older" | age < 30 | 39 |
| 1984 | bank_transactions__adversarial__4 | "older" | customer_age < 45 | 45 |
| 2113 | bank_transactions__array_medium__4 | "older" | customer_age < 30 | 45 |
| 2137 | healthcare__adversarial_slang__4 | "older" | age < 39 | 39 |
| 2302 | healthcare__or_mixed_medium__4 | "older" | age < 30 | 39 |
| 2307 | healthcare__or_mixed_hard__4 | "older" | age < 39 | 39 |
| 2524 | bank_transactions__or_hard__5 | "older" | customer_age < 25 | 45 |
| 2548 | gym_members__or_mixed_easy__5 | "older" | age < 25 | 40 |
| 2706 | bank_transactions__or_mixed_easy__5 | "older" | customer_age < 30 | 45 |
| 2766 | healthcare__or_mixed_easy__5 | "older" | age < 39 | 39 |
| 2980 | diabetes__or_medium__4 | "young" | age > 50 | 29 |
| 3053 | olympic_athletes__medium__2 | "older" | age < 20 | 24 |
| 3056 | olympic_athletes__or_mixed_easy__5 | "older" | age < 20 | 24 |
| 3124 | olympic_athletes__array_easy__2 | "older" | age < 20 | 24 |
| 3194 | diabetes__or_mixed_hard__4 | "older" | age < 29 | 29 |

---

## Issue 2: Tautological ORs (59 instances)

Filter ORs ALL values of a categorical field, matching every row. Teaches the model to generate useless conditions.

**Affected fields:**
- `sex` (male OR female): 14 instances
- `brand` (Adidas OR Nike): 10 instances
- `type` (Movie OR TV Show): 10 instances
- `circuit_type` (race OR street): 5 instances
- `gender` (Female OR Male): 7 instances
- `smoker` (yes OR no): 4 instances
- `transaction_type` (Credit OR Debit): 4 instances
- `hours` (Full time OR Part time): 4 instances
- `furnishing` (Furnished OR Unfurnished): 1 instance
- `season` (Summer OR Winter): 1 instance
- `host_identity_verified`: 1 instance

---

## Issue 3: Wrong Operator for Column Type (12 instances)

| # | Sample | Column | Type | Used | Should use |
|---|--------|--------|------|------|-----------|
| 275 | steam__mix__0 | genres | array | NOT IN | NOT CONTAINS |
| 699 | clothes__array_medium__1 | size | categorical | CONTAINS_ALL | == or IN |
| 923 | board_games__mix__2 | domains | array | NOT IN | NOT CONTAINS |
| 978 | steam__mix__1 | genres | array | NOT IN | NOT CONTAINS |
| 1201 | board_games__negative__2 | domains | array | NOT IN | NOT CONTAINS |
| 1253 | travel__array_medium__2 | language | categorical | CONTAINS | == or IN |
| 1747 | imdb__mix__3 | genre | array | NOT IN | NOT CONTAINS |
| 2178 | steam__mix__4 | genres | array | NOT IN | NOT CONTAINS |
| 2287 | imdb__negative__4 | genre | array | NOT IN | NOT CONTAINS |
| 2364 | travel__array_medium__4 | language | categorical | CONTAINS | == or IN |
| 2443 | board_games__adversarial_mixed__5 | domains | array | NOT IN | NOT CONTAINS |
| 2794 | steam__mix__5 | genres | array | NOT IN | NOT CONTAINS |

---

## Issue 4: False EMPTY (11 instances)

Valid queries marked as EMPTY when schema fields exist.

| # | Sample | Query mentions | Valid field? |
|---|--------|---------------|-------------|
| 239 | healthcare__hallucination_full__0 | "age" | age exists ✓ |
| 335 | dubai__hallucination_full__0 | "rent" | rent exists ✓ |
| 1211 | healthcare__hallucination_full__2 | "age" | age exists ✓ |
| 1353 | dubai__hallucination_full__2 | "rent" | rent exists ✓ |
| 2007 | healthcare__hallucination_full__4 | "age" | age exists ✓ |
| 2610 | healthcare__hallucination_full__5 | "age" | age exists ✓ |
| 3027 | diamonds__hallucination_full__5 | "x" | x exists ✓ |
| 3028 | diamonds__hallucination_full__4 | "x" | x exists ✓ |
| 3101 | olympic__hallucination_full__2 | "medal" | medal exists ✓ |
| 3102 | olympic__hallucination_full__5 | "medal" | medal exists ✓ |
| 3190 | olympic__adversarial_nums__3 | "noc" | noc exists ✓ |

**Note:** Some of these may be correct — the query uses a field name coincidentally but in a context that refers to something not in the schema (e.g., "age of the wine" vs "age" column). Manual review needed.

---

## Issue 5: Impossible Filters (11 instances)

Filter threshold exceeds the column's min/max range — will return zero results.

| # | Sample | Filter | Range |
|---|--------|--------|-------|
| 484 | amazon__hard__1 | list_price > 999.99 | max=999.99 |
| 543 | video_games__array_hard__0 | global_sales > 82.74 | max=82.74 |
| 814 | video_games__adversarial_mixed__1 | na_sales > 41.49 | max=41.49 |
| 883 | video_games__hard__1 | global_sales > 82.74 | max=82.74 |
| 1332 | video_games__array_easy__2 | jp_sales < 0.0 | min=0.0 |
| 1418 | amazon__negative__3 | list_price < 0.0 | min=0.0 |
| 1442 | video_games__array_hard__2 | jp_sales < 0.0 | min=0.0 |
| 1970 | video_games__hard__3 | jp_sales < 0.0 | min=0.0 |
| 2147 | flavors__or_hard__4 | review_date < 2006 | min=2006 |
| 2414 | board_games__adversarial__5 | owned_users > 500000 | max=155312 |
| 3011 | diamonds__adversarial_nums__0 | z > 62 | max=31.8 |

---

## Issue 6: Missing Parentheses (16 instances)

Mixed AND/OR without proper grouping — ambiguous evaluation order.

All 16 instances listed in data_issues.json with full filter text.

---

## Issue 7: Redundant ORs (5 instances)

| # | Sample | Filter | Problem |
|---|--------|--------|---------|
| 40 | amazon__or_easy__0 | bought > 1000 OR bought > 5000 | >5000 is subset of >1000 |
| 310 | gym__or_medium__0 | fat > 26.2 OR fat > 32.55 | >32.55 is subset of >26.2 |
| 672 | board_games__or_easy__1 | bgg_rank < 100 OR bgg_rank < 500 | <100 is subset of <500 |
| 1117 | board_games__or_easy__2 | complexity < 1.97 OR complexity <= 2.36 | <1.97 is subset of <=2.36 |
| 2407 | steam__or_mixed_easy__4 | playtime > 0 OR playtime > 0 | Exact duplicate |

---

## Issue 8: OR Instead of IN (551 instances)

Same-field multi-value comparisons using OR instead of the IN operator convention. Not semantically wrong but inconsistent with the system prompt rule "Use IN for multiple values of the same field."

**Most affected schemas:** adidas_vs_nike, healthcare_insurance, clothes, bank_transactions, gym_members

---

## Recommendations

### Drop immediately (critical — teaches wrong mappings):
- **35 direction contradictions** — "older" → age < median
- **11 impossible filters** — threshold exceeds column range
- **5 redundant ORs** — logically meaningless

### Fix in data generation (prevent future issues):
- **59 tautological ORs** — add validation: if OR covers all values, drop the condition
- **12 wrong operators** — add rule: NOT IN on arrays → NOT CONTAINS
- **16 missing parentheses** — enforce parentheses in mixed AND/OR
- **551 OR instead of IN** — enforce IN convention more strictly in prompts

### Manual review:
- **11 false EMPTYs** — verify if query truly references non-schema concepts

### Total samples to drop: ~51 (1.5% of dataset)
### Total samples to fix via regeneration: ~638 (19.4% of dataset)
