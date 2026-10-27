"""Turn a model's free-text reply into an answer and grade it against the reference."""
import re
from fractions import Fraction

# MCQ: an explicit "Answer: B" / "answer is (B)" wins; otherwise a reply that starts with the letter.
_MCQ_EXPLICIT = re.compile(r"answer\s*(?:is)?\s*[:\-]?\s*\(?([ABCD])\b", re.I)
_MCQ_LEADING = re.compile(r"^\W*\(?([ABCD])(?:[).:\s]|$)")
# Math: the number after "Answer:", else the last number in the reply.
_MATH_EXPLICIT = re.compile(r"answer\s*(?:is)?\s*[:\-]?\s*\$?\s*(-?[\d,]*\.?\d+)", re.I)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_mcq(reply: str) -> str | None:
    match = _MCQ_EXPLICIT.search(reply) or _MCQ_LEADING.search(reply.strip())
    return match.group(1).upper() if match else None


def extract_number(reply: str) -> str | None:
    explicit = _MATH_EXPLICIT.findall(reply)
    candidates = explicit or _NUMBER.findall(reply)
    return candidates[-1].replace(",", "") if candidates else None


def extract_answer(task_type: str, reply: str) -> str | None:
    return extract_mcq(reply) if task_type == "mcq" else extract_number(reply)


def is_correct(task_type: str, reply: str, reference: str) -> bool:
    answer = extract_answer(task_type, reply)
    if answer is None:
        return False
    if task_type == "mcq":
        return answer == reference.strip().upper()
    try:  # compare as exact numbers so "18", "18.0" and "18.00" all match
        return Fraction(answer) == Fraction(reference.replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return False
