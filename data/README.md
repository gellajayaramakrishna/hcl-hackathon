# Stand-in query dataset

**This is a stand-in.** It will be replaced by the organizers' 500 queries and reference
answers. It has the same shape, so the router pipeline can be built and tested now. When the
real file arrives, load it with `src/data_loader.load_queries()`. If its column names differ,
edit only `COLUMN_MAP` in `src/data_loader.py`.

Rebuild with `python scripts/build_dataset.py`. The seed is 42, so every run gives identical files.

## How it was built

| Part | Count | Source | Notes |
|---|---|---|---|
| MMLU easy | 125 | `cais/mmlu` (all, test) | 7 subjects (elementary_mathematics, high_school_geography, high_school_biology, miscellaneous, high_school_us_history, nutrition, marketing), spread evenly (17-18 each) |
| MMLU hard | 125 | `cais/mmlu` (all, test) | 8 subjects (professional_law, college_physics, abstract_algebra, college_mathematics, professional_medicine, formal_logic, college_chemistry, econometrics), 15-16 each |
| GSM8K | 200 | `openai/gsm8k` (main, test) | 67 easy / 67 medium / 66 hard. Difficulty = number of arithmetic operations in the solution's `<<expr=result>>` steps: 1-2 easy, 3-4 medium, 5+ hard. The 31 problems with no countable operations are not sampled |
| Paraphrases | 20 | derived | 10 MMLU + 10 GSM8K. Template rewrites of the wording only. Numbers, entities and options are unchanged. Cache **should reuse** |
| Number-changed | 20 | derived | GSM8K with one number changed. The answer is recomputed by replaying the solution's `<<expr=result>>` steps with a safe evaluator. Cache **must not reuse** |
| Near-duplicates | 10 | derived | 5 MMLU + 5 GSM8K. Only case, spacing or trailing punctuation changed. Cache **should reuse** |

Number-changed items are recomputed by replaying the arithmetic, so a condition in the problem
("an extra point if they have over 20") is not re-checked after the change. All 20 current items
were checked by hand; re-check them if you change the seed or the sampling.

GSM8K test item 1309 is excluded: its official solution is wrong (labelled 2280; the correct answer is 2180).

## Files

- `queries.csv` / `queries.jsonl`: the same 500 rows in two formats.
- `splits.json`: `{"train": [...], "tune": [...], "test": [...]}` lists of ids. The split is
  60/20/20, stratified by task_type × difficulty. Each derived item is in the same split as its original.

The rows are shuffled, but every derived item comes **after** its original. A cache that
processes the file in order has therefore always seen the original first.

## Columns

| Column | Meaning |
|---|---|
| `id` | `q0001`…`q0500`, in file order |
| `query` | Text sent to the model. MCQ format: the question, then `A) …` to `D) …` on separate lines |
| `reference_answer` | MCQ: a letter `A`-`D`. Math: a number string without commas, e.g. `1250` |
| `task_type` | `mcq` or `math` |
| `difficulty` | **Proxy** label set at build time: `easy` / `hard` (MMLU, by subject) or `easy` / `medium` / `hard` (GSM8K, by operation count). Used for balanced sampling and splits only |
| `subject` | MMLU subject, or `gsm8k` |
| `source` | `mmlu`, `gsm8k` or `derived` |
| `solution` | GSM8K worked solution, for analysis only. **Never show it to the models.** For number-changed items it holds the recomputed steps |
| `cache_group` | For derived items, the id of the original query. Empty for base items |
| `cache_expected` | `reuse`, `no_reuse`, or `n/a` (base items) |

## Difficulty: proxy vs empirical

The `difficulty` column is only a **proxy**. It is a structural guess that is available before
any model has run. The real definition is **empirical**: which models fail the query.
`scripts/empirical_difficulty.py` computes it from `model_runs.jsonl` into
`empirical_difficulty.csv` (with `easy`/`medium`/`hard` and a continuous `fail_rate`). It also
reports how well the proxy agrees with it. The organizers' file will have no worked solutions,
so the empirical label is the only one that works there too. See `src/difficulty.py`.
