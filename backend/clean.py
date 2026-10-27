# Step 1: clean a query with rules: normalize, mask PII, and extract the facts the cache
# safety rules and the router need (numbers, names, negation, options, time words, task type).
import re

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?<!\d)(?:\+\d{1,3}[\s-]?)?(?:\d{3}[\s-]?\d{3}[\s-]?\d{4}|\d{5}[\s-]?\d{5})(?!\d)")
NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
NAME = re.compile(r"\b[A-Z][a-z]+\b")
NEGATION = re.compile(r"\b(not|never|except|no)\b|n't\b", re.I)
OPTION = re.compile(r"^\s*\(?([A-D])[).]\s*(.+?)\s*$", re.M)
TIME_WORDS = re.compile(r"\b(today|current|currently|latest|now|this year)\b", re.I)
MATH_WORDS = ["how many", "how much", "total", "sum", "each", "per", "percent", "%", "average",
              "times", "twice", "half", "cost", "price", "profit", "remaining", "left", "more than",
              "less than", "+", "-", "*", "/", "="]
# capitalized words that are not names
NOT_NAMES = {"The", "A", "An", "What", "Which", "Who", "How", "If", "In", "On", "At", "For",
             "Of", "And", "But", "Or", "It", "This", "That", "These", "Those", "He", "She",
             "They", "We", "You", "I", "Is", "Are", "Was", "Were", "When", "Where", "Why",
             "Each", "Every", "After", "Before", "Then", "So", "Final", "Answer", "Statement"}


def _names(text):
    """Capitalized words that do not start a sentence or line."""
    names = set()
    for m in NAME.finditer(text):
        before = text[:m.start()].rstrip(" \t")
        if before and before[-1] not in ".?!:\n" and m.group() not in NOT_NAMES:
            names.add(m.group().lower())
    return sorted(names)


def clean(query, task_type=None):
    redacted = EMAIL.sub("[EMAIL]", query)
    redacted = PHONE.sub("[PHONE]", redacted)
    has_pii = redacted != query
    text = " ".join(redacted.lower().split())

    options = [opt.lower() for _, opt in OPTION.findall(redacted)]
    math_words = sum(text.count(w) for w in MATH_WORDS)
    numbers = sorted({n.replace(",", "") for n in NUMBER.findall(redacted)})
    if task_type is None:  # guess when the caller does not know (e.g. the chat UI)
        task_type = "mcq" if len(options) >= 2 else "math" if numbers and math_words else "text"

    return {
        "query": query,          # original
        "redacted": redacted,    # original case, PII masked: this is what the LLM sees
        "text": text,            # lowercased + trimmed: used for cache key and embedding
        "has_pii": has_pii,
        "numbers": numbers,
        "names": _names(redacted),
        "negation": bool(NEGATION.search(redacted)),
        "options": options,
        "time_sensitive": bool(TIME_WORDS.search(redacted)),
        "math_words": math_words,
        "task_type": task_type,
    }


if __name__ == "__main__":
    for q in ["Tom has 3 apples and buys 5 more. How many apples does Tom have?",
              "Email me at a.b@x.com or call 98765 43210. What is the latest iPhone?",
              "Which of these is NOT a planet?\nA) Mars\nB) Pluto\nC) Venus\nD) Earth"]:
        print(clean(q), "\n")
