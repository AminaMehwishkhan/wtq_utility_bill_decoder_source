"""Deterministic arithmetic for Level 2.

The LLM never has to do maths in its head: it writes small expressions
(``total_taxes / current_bill * 100``) and Python evaluates them with a
restricted AST evaluator (no attribute access, no imports, no arbitrary calls).
"""
from __future__ import annotations

import ast
import math
import operator
import re
import statistics
from typing import Any

MAX_EXPR_LEN = 400


class CalcError(ValueError):
    """Raised for any expression that cannot be evaluated safely and exactly."""


# ----------------------------------------------------------------------- helpers
def _flat(args: tuple) -> list[float]:
    out: list[float] = []
    for a in args:
        if isinstance(a, (list, tuple)):
            out.extend(x for x in a if x is not None)
        elif a is not None:
            out.append(a)
    return out


def _need(vals: list[float], name: str) -> list[float]:
    if not vals:
        raise CalcError(f"{name}() has no values to work with")
    return vals


def _count(op):
    def f(values, threshold):
        return sum(1 for v in _flat((values,)) if op(v, threshold))
    return f


def _pct(part, whole):
    if whole == 0:
        raise CalcError("percentage with zero denominator")
    return part / whole * 100


def _pct_change(old, new):
    if old == 0:
        raise CalcError("percentage change from zero")
    return (new - old) / abs(old) * 100


FUNCS: dict[str, Any] = {
    "sum": lambda *a: sum(_flat(a)),
    "mean": lambda *a: statistics.fmean(_need(_flat(a), "mean")),
    "avg": lambda *a: statistics.fmean(_need(_flat(a), "avg")),
    "median": lambda *a: statistics.median(_need(_flat(a), "median")),
    "max": lambda *a: max(_need(_flat(a), "max")),
    "min": lambda *a: min(_need(_flat(a), "min")),
    "len": lambda *a: len(_flat(a)),
    "abs": abs,
    "round": lambda x, n=0: round(x, int(n)),
    "count_gt": _count(operator.gt),
    "count_ge": _count(operator.ge),
    "count_lt": _count(operator.lt),
    "count_le": _count(operator.le),
    "count_eq": _count(operator.eq),
    "pct": _pct,
    "pct_change": _pct_change,
}

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def evaluate(expr: str, env: dict[str, Any] | None = None) -> float:
    """Evaluate ``expr`` safely. Names must exist in ``env``; lists may be passed to
    sum/mean/max/min/len/count_*. Raises CalcError on anything unsafe or undefined."""
    env = env or {}
    if not isinstance(expr, str) or not expr.strip():
        raise CalcError("empty expression")
    if len(expr) > MAX_EXPR_LEN:
        raise CalcError("expression too long")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise CalcError(f"syntax error: {e.msg}") from None
    result = _eval(tree.body, env)
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise CalcError("expression did not produce a number")
    if isinstance(result, float) and not math.isfinite(result):
        raise CalcError("non-finite result")
    return result


def _eval(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise CalcError("only numeric constants are allowed")
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise CalcError(f"unknown name '{node.id}'")
        val = env[node.id]
        if val is None:
            raise CalcError(f"'{node.id}' is not available on this bill")
        return val
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e, env) for e in node.elts]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand, env))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left, env), _eval(node.right, env)
        if isinstance(node.op, ast.Pow) and (abs(right) > 8 or abs(left) > 1e6):
            raise CalcError("exponent too large")
        try:
            return _BIN[type(node.op)](left, right)
        except ZeroDivisionError:
            raise CalcError("division by zero") from None
        except TypeError:
            raise CalcError("invalid operands") from None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in FUNCS and not node.keywords:
        args = [_eval(a, env) for a in node.args]
        try:
            return FUNCS[node.func.id](*args)
        except CalcError:
            raise
        except (TypeError, ValueError, ZeroDivisionError, statistics.StatisticsError) as e:
            raise CalcError(f"{node.func.id}(): {e}") from None
    raise CalcError(f"unsupported syntax: {type(node).__name__}")


# ----------------------------------------------------------------------- formatting
def format_value(value: float, fmt: str = "plain", decimals: int | None = None) -> str:
    """fmt: plain | pkr | percent | int."""
    whole = abs(value - round(value)) < 1e-9
    if fmt == "int":
        return f"{round(value):,}"
    if fmt == "pkr":
        d = decimals if decimals is not None else (0 if whole else 2)
        return f"PKR {value:,.{d}f}"
    if fmt == "percent":
        d = 1 if decimals is None else decimals
        return f"{value:.{d}f}%"
    d = decimals if decimals is not None else (0 if whole else 2)
    return f"{value:,.{d}f}"


_PLACEHOLDER = re.compile(r"\{([A-Za-z_]\w*)\}")


def render_answer(answer: str, calcs: Any, env: dict[str, Any]) -> str:
    """Evaluate each calc, then substitute ``{id}`` placeholders in ``answer``.
    Raises CalcError when a calc fails or a placeholder stays unresolved."""
    values: dict[str, str] = {}
    for c in calcs or []:
        if not isinstance(c, dict) or "id" not in c or "expr" not in c:
            raise CalcError(f"malformed calc entry {c!r}")
        dec = c.get("decimals")
        dec = int(dec) if isinstance(dec, (int, float)) and not isinstance(dec, bool) else None
        values[str(c["id"])] = format_value(evaluate(str(c["expr"]), env),
                                            str(c.get("format") or "plain"), dec)
    missing = [m for m in _PLACEHOLDER.findall(answer) if m not in values]
    if missing:
        raise CalcError(f"unresolved placeholders {missing}")
    rendered = _PLACEHOLDER.sub(lambda m: values[m.group(1)], answer)
    rendered = re.sub(r"(?i)\bPKR\s+(?=PKR\s)", "", rendered)
    rendered = re.sub(r"%{2,}", "%", rendered)
    return rendered


# ----------------------------------------------------------------------- bill facts
def build_env(level1: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
    """Variables the model may use inside expressions."""
    env: dict[str, Any] = {}
    for k, v in level1.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            env[k] = v
    for kind, key in (("charge", "charges"), ("tax", "taxes")):
        items = level1.get(key) or []
        env[f"{kind}_amounts"] = [i["amount"] for i in items]
        for typ in {i["type"] for i in items}:
            env[f"{kind}_{typ}_total"] = round(sum(i["amount"] for i in items if i["type"] == typ), 2)
    rows = sorted((h for h in history if isinstance(h, dict) and h.get("month")),
                  key=lambda h: h["month"])
    prev = [h for h in rows if not h.get("is_current_month")]
    env["units_prev"] = [h.get("units") for h in prev if h.get("units") is not None]
    env["units_all"] = [h.get("units") for h in rows if h.get("units") is not None]
    env["billed_hist"] = [h.get("billed_amount") for h in rows if h.get("billed_amount") is not None]
    env["paid_hist"] = [h.get("payment") for h in rows if h.get("payment") is not None]
    return env


def derived_facts(level1: dict[str, Any]) -> dict[str, float]:
    """Pre-computed, clearly labelled ratios the model can quote verbatim."""
    out: dict[str, float] = {}
    cb, tt, tc = level1.get("current_bill"), level1.get("total_taxes"), level1.get("total_charges")
    units = level1.get("units_consumed")
    pw, pa = level1.get("payable_within_due_date"), level1.get("payable_after_due_date")
    if cb and tt is not None:
        out["taxes_as_percent_of_current_bill"] = round(tt / cb * 100, 2)
    if cb and tc is not None:
        out["charges_as_percent_of_current_bill"] = round(tc / cb * 100, 2)
    if cb and units:
        out["current_bill_per_unit"] = round(cb / units, 2)
    if pw is not None and pa is not None:
        out["late_payment_extra_vs_within_due_date"] = round(pa - pw, 2)
    return out
