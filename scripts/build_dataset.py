"""Build the stand-in query set for the Cost-Aware Model Router.

Produces 500 queries in the same shape as the organizers' future file:
  * 250 MMLU multiple-choice questions (125 easy subjects, 125 hard subjects)
  * 200 GSM8K math word problems (easy / medium / hard by number of arithmetic operations)
  *  50 cache-test queries derived from the above:
       20 paraphrases        -> cache should REUSE the original's answer
       20 number-changed     -> cache must NOT reuse (the answer is different)
       10 near-duplicates    -> cache should REUSE

Outputs: data/queries.csv, data/queries.jsonl, data/splits.json
Run:     python scripts/build_dataset.py
"""
import json
import random
import re
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_loader import COLUMN_ORDER, load_queries  # noqa: E402
from src.difficulty import ANNOTATION, proxy_difficulty  # noqa: E402
from src.safe_arith import NUMBER_TOKEN, format_number, safe_eval  # noqa: E402

SEED = 42
DATA_DIR = ROOT / "data"

EASY_SUBJECTS = ["elementary_mathematics", "high_school_geography", "high_school_biology",
                 "miscellaneous", "high_school_us_history", "nutrition", "marketing"]
HARD_SUBJECTS = ["professional_law", "college_physics", "abstract_algebra", "college_mathematics",
                 "professional_medicine", "formal_logic", "college_chemistry", "econometrics"]
N_MMLU_PER_LEVEL = 125
N_GSM8K = {"easy": 67, "medium": 67, "hard": 66}
N_PARAPHRASE = {"mcq": 10, "math": 10}
N_NUMBER_CHANGED = 20
N_DUPLICATE = {"mcq": 5, "math": 5}
SPLIT_FRACTIONS = {"train": 0.6, "tune": 0.2, "test": 0.2}
LETTERS = "ABCD"
# GSM8K test items whose reference solution is wrong, found by manual review; never sampled.
# 1309: the total adds Maryam's "$400 more" instead of Sarah's $300 (labelled 2280, correct 2180).
GSM8K_EXCLUDE = {1309}

# Items we tried to use for a cache test but had to skip, with the reason.
SKIPPED: list[tuple[str, str, str]] = []  # (kind, original uid, reason)


def rng_for(stage: str) -> random.Random:
    """A separate seeded RNG per stage, so changing one stage never shifts another's samples."""
    return random.Random(f"{SEED}-{stage}")


def make_row(uid, query, answer, task_type, difficulty, subject, source, solution="",
             kind="base", group="", expected="n/a", original_query=""):
    """One dataset row. `uid` is an internal key; public ids (q0001...) are assigned after shuffling.
    Columns starting with '_' are internal bookkeeping and are dropped before writing."""
    return {"_uid": uid, "query": query, "reference_answer": answer, "task_type": task_type,
            "difficulty": difficulty, "subject": subject, "source": source, "solution": solution,
            "_kind": kind, "_group_uid": group, "cache_expected": expected,
            "_original_query": original_query}


# --------------------------------------------------------------------------- 1-2. MMLU
def format_mcq(question: str, choices: list[str]) -> str:
    options = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices))
    return f"{question}\n{options}"


def stratified_counts(total: int, subjects: list[str]) -> dict[str, int]:
    """Split `total` as evenly as possible; the first subjects get the remainder (125/7 -> 18,18,...,17)."""
    base, extra = divmod(total, len(subjects))
    return {s: base + (1 if i < extra else 0) for i, s in enumerate(subjects)}


def sample_mmlu() -> list[dict]:
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    by_subject = defaultdict(list)
    for idx, subject in enumerate(mmlu["subject"]):
        by_subject[subject].append(idx)

    rng = rng_for("mmlu")
    rows = []
    for difficulty, subjects in (("easy", EASY_SUBJECTS), ("hard", HARD_SUBJECTS)):
        for subject, n in stratified_counts(N_MMLU_PER_LEVEL, subjects).items():
            for idx in rng.sample(by_subject[subject], n):
                item = mmlu[idx]
                rows.append(make_row(
                    uid=f"mmlu-{idx}",
                    query=format_mcq(item["question"].strip(), item["choices"]),
                    answer=LETTERS[item["answer"]],  # integer index 0-3 -> letter
                    task_type="mcq", difficulty=difficulty, subject=subject, source="mmlu"))
    return rows


# --------------------------------------------------------------------------- 3. GSM8K
def gsm8k_final_answer(solution: str) -> str:
    """The number after '####', commas removed: '#### 1,250' -> '1250'."""
    return solution.split("####")[-1].strip().replace(",", "")


def reasoning_lines(solution: str) -> list[str]:
    return [ln for ln in solution.split("\n") if ln.strip() and not ln.startswith("####")]


def sample_gsm8k() -> list[dict]:
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    by_level = defaultdict(list)
    for idx, solution in enumerate(gsm["answer"]):
        # Difficulty = number of arithmetic operations in the solution (see src/difficulty.py).
        # Problems with no countable operations can't be rated, so they are not sampled.
        level = proxy_difficulty(solution)
        if level and idx not in GSM8K_EXCLUDE:
            by_level[level].append(idx)

    rng = rng_for("gsm8k")
    rows = []
    for difficulty, n in N_GSM8K.items():
        for idx in rng.sample(by_level[difficulty], n):
            item = gsm[idx]
            rows.append(make_row(
                uid=f"gsm8k-{idx}", query=item["question"].strip(),
                answer=gsm8k_final_answer(item["answer"]), task_type="math",
                difficulty=difficulty, subject="gsm8k", source="gsm8k", solution=item["answer"]))
    return rows


# --------------------------------------------------------------------------- 4a. paraphrases
def split_mcq(query: str) -> tuple[str, str]:
    """Separate an MCQ query into its question stem and the 'A) ...' option block."""
    stem, _, options = query.partition("\nA) ")
    return stem, "A) " + options


# Each rule is (pattern, replacement). The first rule that changes the stem is used.
# They only touch question wording -- never numbers, names or the options.
MCQ_PARAPHRASE_RULES = [
    (re.compile(r"^Which of the following (is|are) (.+)\?$", re.S), r"Pick the option that \1 \2."),
    (re.compile(r"\bWhich of the following\b"), "Among the listed options, which"),
    (re.compile(r"\bwhich of the following\b"), "which of the listed options"),
    (re.compile(r"^What is (.+)\?$", re.S), r"Can you tell me what \1 is?"),
    (re.compile(r"^Find (.+)$", re.S), r"Determine \1"),
]

# GSM8K: reword the final question, then move it in front of the context (clause reorder).
MATH_QUESTION_REWRITES = [
    (re.compile(r"^How many\b"), "What number of"),
    (re.compile(r"^How much money\b"), "What amount of money"),
    (re.compile(r"^How much\b"), "What amount"),
    (re.compile(r"^What is\b"), "Can you work out what is"),
]
FINAL_QUESTION = re.compile(r"^(?P<context>.*[.!])\s+(?P<question>[A-Z][^.!?]*\?)\s*$", re.S)


def paraphrase_mcq(query: str) -> str | None:
    stem, options = split_mcq(query)
    for pattern, replacement in MCQ_PARAPHRASE_RULES:
        new_stem = pattern.sub(replacement, stem, count=1)
        if new_stem != stem:
            return f"{new_stem}\n{options}"
    return None


def paraphrase_math(query: str) -> str | None:
    match = FINAL_QUESTION.match(query)
    if not match:
        return None
    question = match["question"]
    for pattern, replacement in MATH_QUESTION_REWRITES:
        if pattern.search(question):
            question = pattern.sub(replacement, question, count=1)
            return f"{question} Here is the situation: {match['context']}"
    return None


def numbers_in(text: str) -> list[str]:
    return sorted(m.replace(",", "") for m in re.findall(r"\d[\d,]*(?:\.\d+)?", text))


# --------------------------------------------------------------------------- 4b. number changes
# A number in the question text: 16, 80,000, 2.5. Lookarounds avoid pieces of times (3:30),
# ordinals (5th), percentages (150% is written 1.5 in solutions) and longer numbers.
QUESTION_NUMBER = re.compile(r"(?<![\d.,:])\d{1,3}(?:,\d{3})+(?![\d:%])|(?<![\d.,:])\d+(?:\.\d+)?(?![\d.,:%]|st|nd|rd|th)")
# Values so common as implicit constants (halving, percent, minutes...) that a step result equal
# to one of them could be confused with the constant in a later step -> treat as ambiguous.
COMMON_CONSTANTS = {Fraction(n) for n in (0, 1, 2, 3, 4, 5, 6, 7, 10, 12, 24, 30, 52, 60, 100, 365, 1000)}
# Spelled-out numbers ("six hours") also appear as digits in solutions, so they count as question values.
NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
NUMBER_WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "hundred": 100, "dozen": 12})


def canonical(text: str) -> Fraction:
    return Fraction(text.replace(",", ""))


def question_values(question: str) -> list[Fraction]:
    """Every number stated in the question, as digits or as a word."""
    values = [canonical(m.group()) for m in QUESTION_NUMBER.finditer(question)]
    words = re.findall(r"[a-z]+", question.lower())
    return values + [Fraction(NUMBER_WORDS[w]) for w in words if w in NUMBER_WORDS]


def recompute_chain(solution: str, old: Fraction, new: Fraction,
                    fixed_values: set[Fraction]) -> tuple[Fraction, Fraction, list[str]]:
    """Replay the solution's <<expr=result>> steps with `old` replaced by `new`.

    `mapping` tracks how values change: it starts as {old: new}; after each step whose
    result changes, {old_result: new_result} is added, so later steps that reuse that
    result pick up the new value. Returns (original final value, new final value, new steps).

    Substitution works on values, so it is only safe if every value means one thing. A changed
    intermediate result must therefore not collide with a number stated in the question
    (`fixed_values`), an earlier step's result, or a common constant -- e.g. a $6,000 tax that
    equals a $6,000 property tax elsewhere. Raises ValueError for any such ambiguity or when the
    chain is not a clean arithmetic chain.
    """
    lines = reasoning_lines(solution)
    if not all(ANNOTATION.search(ln) for ln in lines):
        raise ValueError("a reasoning line has no calculator annotation")

    mapping = {old: new}
    seen_results = set()
    used_old = False
    new_steps = []
    result_old = result_new = None
    for expression, stored in ANNOTATION.findall(solution):
        tokens = [canonical(t) for t in NUMBER_TOKEN.findall(expression)]
        used_old |= old in tokens

        def substitute(m, use_mapping):
            value = canonical(m.group())
            return format_number(mapping.get(value, value) if use_mapping else value)

        original_expr = NUMBER_TOKEN.sub(lambda m: substitute(m, False), expression)
        changed_expr = NUMBER_TOKEN.sub(lambda m: substitute(m, True), expression)
        result_old, result_new = safe_eval(original_expr), safe_eval(changed_expr)

        # Sanity: replaying the ORIGINAL step must reproduce the result written in the solution.
        if result_old != canonical(stored.strip()):
            raise ValueError(f"step {expression}={stored} does not re-evaluate exactly")
        if result_new < 0 or (result_old.denominator == 1 and result_new.denominator != 1):
            raise ValueError(f"step {changed_expr} gives a non-sensible value {result_new}")
        if result_old in mapping:  # a later token with this value could mean either quantity
            raise ValueError(f"intermediate value {result_old} is ambiguous")
        if result_new != result_old:
            if result_old in fixed_values | seen_results | COMMON_CONSTANTS:
                raise ValueError(f"intermediate value {result_old} is ambiguous")
            mapping[result_old] = result_new
        seen_results.add(result_old)
        new_steps.append(f"<<{changed_expr}={format_number(result_new)}>>")

    if not used_old:
        raise ValueError("the changed number is not used in any calculation")
    return result_old, result_new, new_steps


def change_one_number(row: dict, rng: random.Random) -> dict | None:
    """Try to build a number-changed variant of a GSM8K row; None if no safe change exists."""
    question, solution = row["query"], row["solution"]
    final = canonical(row["reference_answer"])
    numbers = list(QUESTION_NUMBER.finditer(question))
    values = [canonical(m.group()) for m in numbers]

    # Candidate numbers: integers that appear once in the question and are not common constants.
    candidates = [m for m, v in zip(numbers, values)
                  if v.denominator == 1 and values.count(v) == 1 and v not in COMMON_CONSTANTS]
    rng.shuffle(candidates)
    last_reason = "no suitable number in the question"
    for match in candidates:
        old = canonical(match.group())
        other_numbers = set(question_values(question)) - {old}
        deltas = [1, 2, 3, -1, -2]
        rng.shuffle(deltas)
        for new in [old + d for d in deltas] + [old * 2]:
            if new <= 0 or new in other_numbers:
                continue
            try:
                last_old, new_answer, steps = recompute_chain(solution, old, new, other_numbers)
            except (ValueError, ZeroDivisionError, SyntaxError) as err:
                last_reason = str(err)
                continue
            # The last calculation must BE the final answer, and the answer must actually change.
            if last_old != final:
                last_reason = "last calculation is not the final answer"
                break
            if new_answer == final:
                last_reason = "new answer equals the old one"
                continue
            if final.denominator == 1 and new_answer.denominator != 1:
                last_reason = "new answer is not an integer"
                continue
            text = f"{new:,}" if "," in match.group() else str(new)
            new_question = question[:match.start()] + text + question[match.end():]
            new_solution = "\n".join(steps) + f"\n#### {format_number(new_answer)}"
            return make_row(
                uid=f"{row['_uid']}-numchg", query=new_question,
                answer=format_number(new_answer), task_type="math",
                difficulty=row["difficulty"], subject=row["subject"], source="derived",
                solution=new_solution, kind="number_changed", group=row["_uid"],
                expected="no_reuse", original_query=question)
    SKIPPED.append(("number_changed", row["_uid"], last_reason))
    return None


# --------------------------------------------------------------------------- 4c. near-duplicates
def near_duplicate(query: str, variant: int) -> str:
    """Apply trivial surface changes (case, spacing, trailing punctuation) to the question stem."""
    stem, options = split_mcq(query) if "\nA) " in query else (query, "")
    if variant % 3 == 0:
        stem = stem.lower() + "  "                       # lowercase + trailing spaces
    elif variant % 3 == 1:
        stem = "  " + stem.replace(" ", "  ", 3)        # leading + doubled spaces
    else:
        stem = stem.rstrip("?.!") + "??"                # changed trailing punctuation
        stem = stem[0].lower() + stem[1:]
    return f"{stem}\n{options}" if options else stem


def normalise(text: str) -> str:
    """What a near-duplicate should be identical to after trivial normalisation."""
    lines = [re.sub(r"[?.!]+\s*$", "", ln.strip()) for ln in text.split("\n")]
    return re.sub(r"[ \t]+", " ", "\n".join(lines)).lower()


# --------------------------------------------------------------------------- 4. assemble derived set
def build_cache_tests(base: list[dict]) -> list[dict]:
    rng = rng_for("cache")
    mcq = [r for r in base if r["task_type"] == "mcq"]
    math = [r for r in base if r["task_type"] == "math"]
    rng.shuffle(mcq)
    rng.shuffle(math)
    used = set()  # each original is used by at most one derived item
    derived = []

    # b) number-changed first: it is the most constrained, so it gets first pick of GSM8K items.
    for row in math:
        if sum(d["_kind"] == "number_changed" for d in derived) == N_NUMBER_CHANGED:
            break
        variant = change_one_number(row, rng)
        if variant:
            derived.append(variant)
            used.add(row["_uid"])

    # a) paraphrases: MCQ split evenly between easy and hard subjects.
    def add_paraphrases(pool, n, fn, kind_label):
        made = 0
        for row in pool:
            if made == n:
                break
            if row["_uid"] in used:
                continue
            new = fn(row["query"])
            if new is None:
                SKIPPED.append((f"paraphrase_{kind_label}", row["_uid"], "no template matched"))
                continue
            derived.append(make_row(
                uid=f"{row['_uid']}-para", query=new, answer=row["reference_answer"],
                task_type=row["task_type"], difficulty=row["difficulty"], subject=row["subject"],
                source="derived", solution=row["solution"], kind="paraphrase", group=row["_uid"],
                expected="reuse", original_query=row["query"]))
            used.add(row["_uid"])
            made += 1
        if made < n:
            raise RuntimeError(f"only {made}/{n} {kind_label} paraphrases could be made")

    half = N_PARAPHRASE["mcq"] // 2
    add_paraphrases([r for r in mcq if r["difficulty"] == "easy"], half, paraphrase_mcq, "mcq")
    add_paraphrases([r for r in mcq if r["difficulty"] == "hard"], N_PARAPHRASE["mcq"] - half,
                    paraphrase_mcq, "mcq")
    add_paraphrases(math, N_PARAPHRASE["math"], paraphrase_math, "math")

    # c) near-duplicates
    variant = 0
    for pool, n in ((mcq, N_DUPLICATE["mcq"]), (math, N_DUPLICATE["math"])):
        for row in [r for r in pool if r["_uid"] not in used][:n]:
            derived.append(make_row(
                uid=f"{row['_uid']}-dup", query=near_duplicate(row["query"], variant),
                answer=row["reference_answer"], task_type=row["task_type"],
                difficulty=row["difficulty"], subject=row["subject"], source="derived",
                solution=row["solution"], kind="duplicate", group=row["_uid"], expected="reuse",
                original_query=row["query"]))
            used.add(row["_uid"])
            variant += 1
    return derived


# --------------------------------------------------------------------------- 5. order, ids, splits
def order_rows(base: list[dict], derived: list[dict]) -> list[dict]:
    """Shuffle, but insert every derived item somewhere AFTER its original."""
    rng = rng_for("order")
    rows = base[:]
    rng.shuffle(rows)
    for item in derived:
        pos = next(i for i, r in enumerate(rows) if r["_uid"] == item["_group_uid"])
        rows.insert(rng.randint(pos + 1, len(rows)), item)
    return rows


def make_splits(rows: list[dict]) -> dict[str, list[str]]:
    """60/20/20 split of base items, stratified by (task_type, difficulty).
    Derived items go to the same split as their original."""
    rng = rng_for("splits")
    strata = defaultdict(list)
    for r in rows:
        if r["_kind"] == "base":
            strata[(r["task_type"], r["difficulty"])].append(r["_uid"])

    split_of = {}
    for key in sorted(strata):
        uids = strata[key]
        rng.shuffle(uids)
        n_train = round(len(uids) * SPLIT_FRACTIONS["train"])
        n_tune = round(len(uids) * SPLIT_FRACTIONS["tune"])
        for i, uid in enumerate(uids):
            split_of[uid] = "train" if i < n_train else "tune" if i < n_train + n_tune else "test"

    splits = {name: [] for name in SPLIT_FRACTIONS}
    for r in rows:
        split = split_of[r["_uid"] if r["_kind"] == "base" else r["_group_uid"]]
        r["_split"] = split
        splits[split].append(r["id"])
    return splits


# --------------------------------------------------------------------------- 7. validation
def is_number(text: str) -> bool:
    try:
        Fraction(text)
        return True
    except (ValueError, ZeroDivisionError):
        return False


def validate(df: pd.DataFrame, splits: dict) -> list[str]:
    errors = []
    by_id = df.set_index("id")
    if len(df) != 500:
        errors.append(f"expected 500 rows, got {len(df)}")
    if not df["id"].is_unique:
        errors.append("ids are not unique")
    for col in ("query", "reference_answer"):
        empty = df[df[col].str.strip() == ""]["id"].tolist()
        if empty:
            errors.append(f"empty {col}: {empty}")

    bad_mcq = df[(df.task_type == "mcq") & ~df.reference_answer.isin(list(LETTERS))]["id"].tolist()
    bad_math = df[(df.task_type == "math") & ~df.reference_answer.map(is_number)]["id"].tolist()
    if bad_mcq:
        errors.append(f"mcq answers not in A-D: {bad_mcq}")
    if bad_math:
        errors.append(f"math answers that are not numbers: {bad_math}")

    position = {qid: i for i, qid in enumerate(df["id"])}
    for _, r in df[df.source == "derived"].iterrows():
        g = r["cache_group"]
        if g not in by_id.index:
            errors.append(f"{r.id}: cache_group {g!r} is not an existing id")
            continue
        orig = by_id.loc[g]
        if position[g] >= position[r.id]:
            errors.append(f"{r.id} appears before its original {g}")
        if r._split != orig._split:
            errors.append(f"{r.id} is in {r._split} but its original {g} is in {orig._split}")
        if r._kind == "number_changed" and r.reference_answer == orig.reference_answer:
            errors.append(f"{r.id}: number-changed answer equals original's")
        if r._kind in ("paraphrase", "duplicate") and r.reference_answer != orig.reference_answer:
            errors.append(f"{r.id}: answer differs from original's")
        if r._kind == "paraphrase" and numbers_in(r.query) != numbers_in(orig.query):
            errors.append(f"{r.id}: paraphrase changed the numbers")
        if r._kind == "duplicate" and normalise(r.query) != normalise(orig.query):
            errors.append(f"{r.id}: near-duplicate differs by more than trivial changes")
    base = df[df.source != "derived"]
    if not ((base.cache_expected == "n/a") & (base.cache_group == "")).all():
        errors.append("base rows must have cache_expected='n/a' and empty cache_group")

    all_split_ids = sum(splits.values(), [])
    if sorted(all_split_ids) != sorted(df["id"]):
        errors.append("splits.json does not cover every id exactly once")
    return errors


def print_summary(df: pd.DataFrame) -> None:
    print("\n=== Counts ===")
    for col in ("source", "task_type", "difficulty", "_split", "cache_expected", "_kind"):
        counts = ", ".join(f"{k}={v}" for k, v in df[col].value_counts().sort_index().items())
        print(f"{col.lstrip('_'):15s} {counts}")
    print("\ndifficulty by task_type:")
    print(pd.crosstab(df.task_type, df.difficulty).to_string())

    print("\n=== Examples (2 per type) ===")
    groups = [("mcq", (df._kind == "base") & (df.task_type == "mcq")),
              ("math", (df._kind == "base") & (df.task_type == "math")),
              ("paraphrase", df._kind == "paraphrase"),
              ("number-changed", df._kind == "number_changed"),
              ("duplicate", df._kind == "duplicate")]
    for label, mask in groups:
        for _, r in df[mask].head(2).iterrows():
            print(f"\n--- {label} | {r.id} | {r.difficulty} | {r.subject} | answer={r.reference_answer}"
                  + (f" | original={r.cache_group} ({r.cache_expected})" if r.cache_group else ""))
            if r._original_query:
                print(f"ORIGINAL: {r._original_query!r}")
                print(f"DERIVED : {r.query!r}")
            else:
                print(r.query)


# --------------------------------------------------------------------------- main
def main() -> None:
    print("Loading and sampling MMLU / GSM8K ...")
    base = sample_mmlu() + sample_gsm8k()
    derived = build_cache_tests(base)
    rows = order_rows(base, derived)

    # Public ids follow file order, so q0001 is the first query the router sees.
    uid_to_id = {r["_uid"]: f"q{i:04d}" for i, r in enumerate(rows, start=1)}
    for r in rows:
        r["id"] = uid_to_id[r["_uid"]]
        r["cache_group"] = uid_to_id[r["_group_uid"]] if r["_group_uid"] else ""
    splits = make_splits(rows)
    df = pd.DataFrame(rows)

    errors = validate(df, splits)
    print_summary(df)

    print("\n=== Skipped while building cache tests ===")
    for kind, uid, reason in SKIPPED:
        print(f"  {kind:16s} {uid:14s} {reason}")
    print(f"  ({len(SKIPPED)} skipped)")

    if errors:
        print("\nVALIDATION FAILED:")
        for e in errors:
            print("  -", e)
        sys.exit(1)

    DATA_DIR.mkdir(exist_ok=True)
    public = df[COLUMN_ORDER]
    public.to_csv(DATA_DIR / "queries.csv", index=False)
    public.to_json(DATA_DIR / "queries.jsonl", orient="records", lines=True, force_ascii=False)
    (DATA_DIR / "splits.json").write_text(json.dumps(splits, indent=2))

    # Round-trip check: the files we wrote must load back identically through the loader.
    for name in ("queries.csv", "queries.jsonl"):
        reloaded = load_queries(DATA_DIR / name)
        if not reloaded.equals(public.astype(str)):
            sys.exit(f"VALIDATION FAILED: {name} does not round-trip through load_queries")
    print(f"\nValidation passed. Wrote {len(public)} rows to {DATA_DIR}/ "
          f"(train={len(splits['train'])}, tune={len(splits['tune'])}, test={len(splits['test'])})")


if __name__ == "__main__":
    main()
