"""Two ways of measuring how hard a query is.

1. PROXY (structural): for GSM8K, the number of arithmetic operations in the worked
   solution's <<expr=result>> steps -- i.e. how many reasoning steps the problem needs.
   Available offline, before any model has run; used to build balanced samples and splits.
   (MMLU has no solutions, so its proxy is the subject list in build_dataset.py.)

2. EMPIRICAL (behavioural, the real definition): which models actually fail the query.
   This is what the router cares about and works on any dataset, including the
   organizers', but only exists after scripts/run_models.py has been run.
"""
import ast
import re

ANNOTATION = re.compile(r"<<([^=<>]+)=([^<>]+)>>")

# Proxy buckets: 1-2 operations = easy, 3-4 = medium, 5+ = hard.
# On the GSM8K test set this gives 344 / 554 / 390 problems.
PROXY_THRESHOLDS = {"easy": 2, "medium": 4}


def count_operations(solution: str) -> int:
    """Number of binary operations (+ - * / **) across all calculator steps.
    `<<16-3-4=9>>` counts 2, `<<9*2=18>>` counts 1. Unparseable steps count 0."""
    total = 0
    for expression, _ in ANNOTATION.findall(solution):
        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except SyntaxError:
            continue
        total += sum(isinstance(node, ast.BinOp) for node in ast.walk(tree))
    return total


def proxy_difficulty(solution: str) -> str | None:
    """easy / medium / hard from the operation count; None if the solution has no operations."""
    n = count_operations(solution)
    if n == 0:
        return None
    if n <= PROXY_THRESHOLDS["easy"]:
        return "easy"
    return "medium" if n <= PROXY_THRESHOLDS["medium"] else "hard"


def empirical_difficulty(correct_by_tier: dict[str, bool], tiers: list[str]) -> str:
    """easy   = the cheapest model gets it right
       medium = the cheapest fails, but some middle-tier model gets it right
       hard   = only the largest model gets it right, or no model does
    `tiers` is ordered cheapest first."""
    cheapest_correct = next((t for t in tiers if correct_by_tier[t]), None)
    if cheapest_correct == tiers[0]:
        return "easy"
    if cheapest_correct in tiers[1:-1]:
        return "medium"
    return "hard"


def fail_rate(correct_by_tier: dict[str, bool]) -> float:
    """Fraction of tiers that got it wrong: a finer-grained, continuous difficulty score."""
    return 1 - sum(correct_by_tier.values()) / len(correct_by_tier)
