"""Safe arithmetic evaluator for GSM8K calculator annotations.

GSM8K solutions contain steps like ``<<16-3-4=9>>``. To recompute an answer
after changing a number we must evaluate those expressions -- but never with
Python's ``eval`` (that would run arbitrary code). Instead we parse the
expression into an AST and only allow numbers, + - * / ** and parentheses.

All arithmetic is done with ``fractions.Fraction`` so results are exact
(0.1 + 0.2 == 0.3), which lets us compare against the stored step results.
"""
import ast
import operator
import re
from fractions import Fraction

# Matches a plain number token: 12, 3.5, .5  (no commas -- annotations don't use them)
NUMBER_TOKEN = re.compile(r"\d+(?:\.\d+)?|\.\d+")

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_MAX_EXPONENT = 10  # guard against huge powers like 9**9**9


def _eval_node(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        # str() first so 0.1 becomes Fraction("0.1"), not the float's binary value
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _eval_node(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if right.denominator != 1 or abs(right) > _MAX_EXPONENT:
                raise ValueError("only small integer exponents are allowed")
            return left ** int(right)
        if type(node.op) in _BINARY_OPS:
            return _BINARY_OPS[type(node.op)](left, right)  # Div by zero raises
    raise ValueError(f"disallowed syntax: {ast.dump(node)}")


def safe_eval(expression: str) -> Fraction:
    """Evaluate an arithmetic expression exactly. Raises ValueError/ZeroDivisionError/SyntaxError."""
    tree = ast.parse(expression.strip(), mode="eval")
    return _eval_node(tree.body)


def format_number(value: Fraction) -> str:
    """Render a Fraction as text: '1250', '2.5', or '(10/3)' if it has no finite decimal form."""
    if value.denominator == 1:
        return str(value.numerator)
    # A fraction has a finite decimal form iff its denominator only has factors 2 and 5
    d = value.denominator
    for p in (2, 5):
        while d % p == 0:
            d //= p
    if d == 1:
        text = f"{float(value):.10f}".rstrip("0").rstrip(".")
        if Fraction(text) == value:
            return text
    return f"({value.numerator}/{value.denominator})"
