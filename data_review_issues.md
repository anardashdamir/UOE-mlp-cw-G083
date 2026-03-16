# Manual Data Review Issues — data_gen.query_variants_reasoning_v10.json

Issues found by reading every sample. Format: [index] issue description.

## Samples 0-99

- [10] Wrong syntax: `'Action' IN genre` should be `genre IN ['Action']`
- [15] `episodes < 2` but query says "nothing with too many episodes" — should be something like `episodes < 50`
- [27] `episodes NOT IN [50, 100, 500]` but query says "don't want anything with more than 50 episodes" — should be `episodes <= 50`
- [51] `bgg_rank >= 5000` but query says "not those ranked below 5000" — higher rank = worse. Should be `bgg_rank <= 5000`
- [52] `text_reviews_count > 49` but query says "> 50 txt reviews" — explicit number overridden by median
- [55] `transaction_duration > 200` but query says "exclude any that took longer than 200 minutes" — should be `transaction_duration < 200`
- [81] `(language_code == 'fre' OR language_code == 'ger')` — should use IN per convention

## Samples 100-199

- [126] "classic earthy tones" → `color IN ['Black', 'White']` — not really earthy tones, minor semantic mismatch
- [132] Redundant price: `price < 42) OR ... AND price < 42) AND price < 50` — < 42 already satisfies < 50
- [134] Query says "cheap stuff under five hundred bucks" (explicit 500) but filter uses `product_price < 1513.02` (median) — should be `product_price < 500`
- [158] TRUNCATED FILTER: `listed_in IN ['Children & Family Movies',` — incomplete, broken. DROP
- [159] `smoker IN ['yes']` — unnecessary IN for single value, should be `smoker == 'yes'`
- [163] `genres IN ['RPG', 'Action']` means ANY but query says "RPG & Action" meaning BOTH — should be `genres IN ['RPG'] AND genres IN ['Action']`
- [182] Query says "laptops" but filter missing `product_category == 'Laptops'`
- [191] `rating NOT IN ['G', 'PG', 'TV-Y']` redundant since already `rating IN ['R', 'TV-MA']`
- [194] `sex IN ['female', 'male']` tautological — matches everyone. Remove condition
- [197] `name LIKE '%three%'` — LIKE operator doesn't exist in our DSL! DROP
- [199] Query says "five bedrooms minimum" but filter has no beds condition — add `beds >= 5`

## Samples 200-299

- [204] `rating != 'NR'` redundant since `rating IN ['R', 'TV-MA']` already excludes NR
- [213] `rent < 100000 AND ... rent < 75000` — redundant, remove `rent < 100000`
- [218] Query says "cheap like under fifty k rent" (explicit 50000), filter uses `rent < 100000` (median) — should be `rent < 50000`
- [242] `platforms == 'linux'` and `genres == 'Action'` — using == on array fields. Should use IN
- [252] Query says "short like under an hour" (explicit < 60 min), filter uses `duration_minutes < 98` (median) — should be `duration_minutes < 60`
- [257] Query says "under 2 mins" (explicit 120 secs), filter uses `transaction_duration < 112` (median) — should be `transaction_duration < 120`
- [260] `listed_in IN ['Crime TV Shows', 'Thrillers'] AND type == 'Movie'` — Crime TV Shows can't be Movie. Logic error
- [264] Query says "more den tree" (three = 3 children), filter uses `children > 1.0` — should be `children > 3`
- [271] Filter missing `type == 'Movie'` — query says "movies" but filter only has duration
- [277] `episodes < 2 OR episodes == 1` — redundant OR (== 1 is subset of < 2)
- [282] Labeled `adversarial_numbers_as_words` but query "Show me all credit transactions" has no numbers as words

## Samples 300-399

- [301] `transaction_type IN ['Credit', 'Debit']` — tautological, matches all values. Remove condition
- [326] `name LIKE '%fantasy%'` — LIKE operator doesn't exist in our DSL! DROP
- [332] `genre IN ['Psychological', 'Horror', 'Thriller', 'Mystery']` — query says "dark and psychological" but filter includes 4 genres as ANY. Acceptable but debatable

## Samples 400-499

- [408] Query says "TH in Dubai" but filter missing `city == 'Dubai'`
- [417] Query says "at least 2 bathes" but filter uses `baths == 2` — should be `baths >= 2`
- [419] "top satisfaction scores" but filter uses `customer_satisfaction >= 3` — should be >= 4
- [433] `furnishing IN ['Furnished', 'Unfurnished']` tautological — remove condition
- [441] `genres IN ['Indie', 'Action', 'Adventure']` means ANY. Query says "indie w/ action or adventure" — should be `genres IN ['Indie'] AND genres IN ['Action', 'Adventure']`
- [447] `brand IN ['Nike', 'Puma'] AND brand != 'Reebok'` — redundant !=
- [457] Missing outer parens on complex OR: depth/table AND binds to second branch only
- [469] Query says "young female" but filter has `age > 39` — should be `age < 39`
- [474] Missing parens in date OR logic
- [475] `type != 'Hotel Apartment'` redundant since `type IN ['Villa', 'Townhouse']`
- [481] Query says "Apartments" but filter missing `type == 'Apartment'`
- [486] Query says "studio or one-bedroom" but `beds <= 2` — should be `beds <= 1`
- [488] `sex IN ['female', 'male']` tautological

## Samples 500-599

- [513] Wrong syntax: `'Strategy' IN genres` — should be `genres IN ['Strategy']`
- [514] Missing `type == 'TV Show'` — query says "TV shows"
- [525] `city == 'Dubai' AND city NOT IN ['Abu Dhabi']` — redundant NOT IN
- [539] Query says "over one carat" but filter uses `carat > 0.7` (median) — should be `carat > 1`
- [553] Query mentions "she" (female) but filter missing `sex == 'female'`
- [556] Query says "depth greater than 62" but filter uses `z > 3.53` — wrong field
- [557] `brand == 'Nike' AND brand != 'Under Armour'` — redundant !=
- [578] Query says "young female" but filter has `age > 39` — should be `age < 39`
- [580] `publisher == 'Penguin' AND publisher != 'Random House'` — redundant !=
- [582] `listed_in IN ['Dramas', 'International Movies']` means ANY but query says BOTH — should split
- [596] Query says "rating above 7" (explicit) but filter uses `> 6.57` (median)
- [599] `charges < 9382.03 AND charges < 5000` — redundant, keep < 5000 only

## Samples 600-699

- [608] "both dry and oily" → `dry == true AND oily == true` — should probably be `dry_and_oily == true`
- [636] Missing `type == 'TV Show'` — query says "shows" with seasons
- [638] `name LIKE '%two%'` — LIKE doesn't exist in DSL! DROP
- [643] Missing `state == 'ny'` — query says "in New York"
- [647] Query says "more then two hundred users" but filter uses `users_rated > 120` (median) — should be > 200
- [659] Query says "more than three" children but filter uses `children > 1.0` — should be > 3
- [660] Wrong syntax: `platforms NOT IN ['windows'] AND 'linux' IN platforms` — reversed
- [661] `genre == 'Action' OR genre == 'Romance'` — should use IN
- [684] `genres IN ['Action', 'Indie']` means ANY but query says "Action AND Indie" — should split

## Samples 700-799

- [706] Tautological OR removed but query still references both types
- [721] "FC" → Face Mask is questionable abbreviation
- [725] Missing `category == 'Shoes'` — "kicks" means shoes
- [726] Missing `publisher == 'Penguin'` — query mentions it, it's valid
- [727] `language_code IN ['fre']` — unnecessary IN for single value
- [728] Query says "BMI over 30" (explicit) but filter uses `> 30.4` (median)
- [765] Missing parens in complex OR
- [786] "excellent" satisfaction but filter uses `>= 3` — should be >= 4
- [794] Wrong syntax: `'Action' IN genres` — should be `genres IN ['Action']`

## Samples 800-949

- [803] `domains == 'Abstract Games'` — using == on array, should use IN
- [820] Query says "May 1, 2019" but filter has `'2019-05-09'` — wrong date
- [824] `city IN [...] AND city != 'Abu Dhabi'` — redundant !=
- [835] `rent < 100000 AND rent < 100000` duplicate. Query says "under 70k" but filter uses 100000
- [854] `genres IN ['Indie', 'RPG']` means ANY but should be BOTH
- [880] Query says "more than 1500" but filter uses `> 1513.02` (median)
- [883] `city IN [...] AND city != 'Sharjah'` — redundant !=
- [891] `domains ==` on array field — should use IN
- [897] Query says "five hundred k" (500000) but filter uses `> 120` — WRONG!
- [900] `brand IN [...] AND brand NOT IN ['Other Brands']` — redundant
- [903] `publisher == 'Penguin Press'` — might not be valid value
- [907] `release_date > '2015'` — bare year, should be '2015-01-01'
- [935] `transaction_type IN ['Credit', 'Debit']` — tautological
- [944] Query says "under forty" (explicit 40) but filter uses `< 42` (median)

## Samples 950-1099

- [955] `brand LIKE '%Glow%'` — LIKE doesn't exist! DROP
- [956] `genres IN ['RPG', 'Strategy']` means ANY but query says BOTH
- [966] Missing `sensitive == true` — query says "sensitive skin"
- [967] `age > 30 OR age <= 55` matches almost everyone. Should be `age < 30 OR age >= 55`
- [1003] "combination skin" → `oily == true AND dry == true` — should be `dry_and_oily == true`
- [1010] Reversed syntax: `'Violent' IN genres`
- [1015] Same as [967] — wrong age logic
- [1023] Missing `dogs_allowed == false` — query says "no dogs"
- [1025] `product_category != 'Tablets'` redundant
- [1040] Filter missing `genres IN ['Action'] AND platforms IN ['windows']`
- [1048] `listed_in == 'Sci-Fi & Fantasy'` — == on array, should use IN
- [1068] Missing `category == 'Jacket'` — query says "jacket"
- [1080] `type != 'OVA'` redundant since `type IN ['TV', 'Movie']`
- [1083] `listed_in IN ['Dramas', 'International Movies']` means ANY but query says BOTH

## Samples 1100-2390 (automated scan)

### LIKE operator (DROP these):
- [1111], [1433]

### == on array field (should use IN):
- [1131], [1227], [1347], [1565], [1597], [1854], [1907], [1993], [2004], [2126], [2207], [2234], [2266]

### Reversed IN syntax ('val' IN col instead of col IN ['val']):
- [1341], [1433], [1646], [1926], [2013]

### Tautological conditions:
- [1255] sex, [1293] transaction_type, [1636] sex, [1915] sex, [1925] sex, [2033] transaction_type, [2248] sex

### Duplicate conditions (same field < X appears twice):
- [1139], [1206], [1422], [1455], [1698], [1744], [1982], [2049], [2055], [2065], [2109], [2241], [2243]

### Redundant != with IN:
- [1151], [1178], [1420]

### ANY vs ALL confusion (IN ['A', 'B'] but query says A AND B):
- [1173], [1260], [1743], [1797], [1827], [1928], [1985], [2048], [2114], [2196], [2217]

---

## GRAND TOTAL

| Issue Type | Count | Action |
|-----------|-------|--------|
| == on array field | ~20 | Fix: change to IN |
| Explicit number overridden by median | ~15 | Fix: use explicit number |
| ANY vs ALL confusion | ~15 | Fix: split into separate IN clauses |
| Duplicate conditions | ~13 | Fix: remove duplicate |
| Tautological OR | ~12 | Fix: remove condition |
| Direction contradiction | ~10 | Fix: flip operator |
| Redundant != with IN or == | ~10 | Fix: remove redundant |
| Missing conditions | ~10 | Fix: add missing field |
| Wrong/reversed syntax | ~8 | Fix: correct syntax |
| LIKE operator | ~5 | DROP sample |
| Missing parens | ~5 | Fix: add parens |
| Truncated filter | ~1 | DROP sample |
| **TOTAL** | **~124** | |

**124 issues out of 2,391 samples = 5.2% error rate**

## Samples 1100-1199 (manual)

- [1100] Missing parens in OR: `type == 'Movie' AND rating == 'R' OR type == 'TV Show'`
- [1111] LIKE operator — DROP
- [1112] Missing parens: satisfaction/frequency conditions only apply to Sony branch
- [1124] Overcomplicated logic — simplify to `furnished OR EV charging`
- [1131] `listed_in ==` on array field
- [1139] Duplicate: `price < 42 AND price < 30`
- [1151] Redundant `city != 'Abu Dhabi'`
- [1152] Query says "over a carat" but filter uses `carat > 0.7` (median)
- [1155] Missing `cut == 'Ideal'`
- [1166] Missing `sensitive == true`
- [1178] Redundant `color != 'F'`
- [1179] "activewear bottoms" → `category IN ['Jeans', 'Shoes']` — Shoes aren't bottoms
- [1199] `brand NOT IN ['budget', 'cheap']` — invalid brand values

## Updated Total: ~140 issues across 2,391 samples (~5.9%)

## Samples 1200-1399 (manual)

- [1204] Explicit 500 → median 1513 for price
- [1220] Explicit 30 → median 30.4 for bmi
- [1227] == on array: genre
- [1243] Missing label == 'Moisturizer'
- [1255] Tautological sex
- [1260] ANY vs ALL: genres Action & Adventure
- [1267] Missing category == 'Shoes'
- [1293] Tautological transaction_type
- [1295] Adds cats_allowed but query only mentions dogs
- [1305] Explicit 30 → median 30.4 for bmi
- [1320] "don't care about brand" but filter has brand == 'Nike'
- [1341] Reversed IN syntax
- [1347] == on array: genre
- [1351] Missing release_year < 2010
- [1355] Tautological type
- [1386] ANY vs ALL: genres Indie+RPG

## Samples 1400-1599 (manual)

- [1400] Missing parens in OR
- [1402] dry + dry_and_oily contradictory
- [1420] Redundant rating !=
- [1433] Reversed IN syntax (3 occurrences)
- [1440] IMPOSSIBLE: num_pages > 302 AND num_pages < 302
- [1457] ANY vs ALL: genres Action+Indie
- [1465] "large group" but min_players <= 2 — contradictory
- [1477] "younger" but age > 39 in first OR branch
- [1480] Explicit < 60 min but filter uses < 98 (median)
- [1484] name == 'RPG' exact match but query says "in the title" — DROP
- [1508] name == 'Action' same issue — DROP
- [1532] ANY vs ALL: Romance+Comedy
- [1537] ANY vs ALL: mac+linux platforms
- [1548] Tautological smoker
- [1565] == on array: listed_in
- [1574] Missing category for "gym clothes"
- [1579] Redundant rank !=
- [1581] Explicit 15000 → median 9382 for charges
- [1590] IMPOSSIBLE: two contradictory cut conditions
- [1597] == on array: genre

## Updated Grand Total: ~185 issues across 2,391 samples (~7.7%)

## Samples 1600-1899 (manual)

- [1602] Adds cats_allowed but query only mentions dogs
- [1629] Missing outer parens on OR
- [1636] Tautological sex
- [1638] "combination skin" → should be dry_and_oily == true
- [1646] Reversed IN syntax
- [1659] Unnecessary IN for single value
- [1664] Duplicate OR conditions (identical)
- [1668] publisher == 'Penguin Publishers' — might not be valid
- [1694] Redundant type NOT IN since == already restricts
- [1696] Duplicate: num_pages > 300 AND num_pages > 302
- [1698] Redundant OR: bgg_rank <= 500 OR bgg_rank <= 1000
- [1715] EMPTY but rank exists in schema — should NOT be EMPTY
- [1722] ANY vs ALL: Action + Romance
- [1728] Missing dogs_allowed for "pet-friendly"
- [1744] Duplicate: price < 2401 AND price < 2401.0
- [1750] "depth greater than 3.53" but filter uses z — wrong field?
- [1782] Missing type == 'Movie'
- [1794] Missing outer parens on complex OR
- [1814] ANY vs ALL: Indie + RPG
- [1827] ANY vs ALL: mac + linux platforms
- [1841] Explicit 9000 → median 9382
- [1843] login_attempts < 1 means 0 — should be <= 1
- [1854] == on array: domains
- [1875] Adds cats_allowed but query only mentions dogs
- [1880] IMPOSSIBLE: price > 42 AND price < 42
- [1895] Missing size == 'M'

## Samples 1900-2024 (manual)

- [1905] "depth" vs z field confusion
- [1907] == on array: genre
- [1908] Query says "three beds" but filter uses beds >= 2
- [1915] Tautological sex
- [1925] Tautological sex
- [1926] Reversed IN syntax
- [1928] ANY vs ALL: platforms windows + mac
- [1990] release_year < 2017.0 — float
- [1993] == on array: listed_in
- [2004] == on array: listed_in
- [2013] Reversed IN syntax

## FINAL TOTAL: ~210 issues across 2,391 samples (8.8%)

### Issue breakdown:
- == on array fields: ~25
- ANY vs ALL confusion: ~20
- Tautological conditions: ~15
- Redundant/duplicate conditions: ~15
- Explicit number → median override: ~15
- Missing conditions from query: ~15
- Reversed IN syntax: ~10
- Missing/wrong parens: ~10
- Direction contradictions: ~10
- LIKE operator (DROP): ~5
- IMPOSSIBLE filters: ~3
- Wrong field (depth vs z): ~3
- Truncated filter: ~1
- Other: ~63

## Samples 2025-2149 (manual)

- [2032] "young Students" but filter has customer_age > 45 — contradicts. Missing parens on OR
- [2033] Tautological transaction_type
- [2042] Missing type == 'apartment'
- [2045] Missing product_category == 'Laptops'
- [2049] Duplicate price conditions
- [2055] Duplicate price conditions
- [2065] Duplicate price, explicit 2000 → median 1513
- [2071] "nineteen seventies" → year_published < 1970 — should be 1970-1979 range
- [2072] Missing outer parens on OR
- [2073] "depth greater than 62" but filter uses z — wrong field
- [2081] customer_age >= 25 AND customer_age >= 35 — should be >= 25 AND <= 35
- [2095] Explicit 30 → median 30.4 for bmi
- [2100] Missing category == 'Shoes' and "no Puma" condition
- [2103] Missing price < 900
- [2114] ANY vs ALL: Action + Adventure
- [2115] Missing genres IN ['Indie']
- [2126] == on array: genre

## Samples 2150-2390 (manual)

- [2154] EMPTY but query has valid genre values — should NOT be EMPTY
- [2167] Missing region exclusion
- [2181] OPPOSITE: text_reviews_count > 100 but query says "don't want more than 100" — should be <= 100
- [2198] "younger male" but age > 39 — should be < 39
- [2200] Explicit 50 → median 108 for price
- [2206] Missing outer parens on complex OR
- [2207] == on array: listed_in
- [2217] ANY vs ALL: platforms linux + mac
- [2224] Missing location == 'Dubai Marina'
- [2234] == on array: domains
- [2241] Query says "under 300k" but filter uses rent < 100000
- [2243] Duplicate price conditions
- [2248] Tautological sex
- [2266] == on array: listed_in
- [2268] IMPOSSIBLE: language can't be eng/fre AND ger/spa simultaneously
- [2275] Duplicate price conditions
- [2289] Missing label == 'Moisturizer'
- [2290] Missing platforms condition
- [2296] "young females" but age > 39
- [2308] Explicit 30 → median 30.4 for bmi
- [2309] Missing type == 'apartment'
- [2311] IMPOSSIBLE: median_playtime < 0
- [2324] Explicit > 12 but filter uses > 10
- [2331] Missing city == 'Dubai'
- [2336] Missing outer parens on OR
- [2352] Missing outer parens on OR
- [2354] SYNTAX ERROR: missing closing bracket

## FINAL GRAND TOTAL: ~240 issues across 2,391 samples (10.0%)

All 2,391 samples manually reviewed.
