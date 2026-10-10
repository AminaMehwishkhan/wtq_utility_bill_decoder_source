"""Level 1 schema, normalisation and validation.

Nothing here invents data: unreadable or malformed values become ``None``.
Printed amounts are never recalculated or rounded.
"""
from __future__ import annotations

import json
import math
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

PROVIDERS = ("KE", "LESCO", "IESCO")
CHARGE_TYPES = ("energy", "fixed", "fpa", "quarterly_adjustment", "surcharge",
                "meter_rent", "subsidy", "other")
TAX_TYPES = ("gst", "electricity_duty", "income_tax", "municipal_tax", "other_tax")

LEVEL1_KEYS = (
    "provider", "tariff", "sanctioned_load_kw", "bill_month", "reading_date",
    "issue_date", "due_date", "previous_reading", "current_reading",
    "units_consumed", "charges", "total_charges", "taxes", "total_taxes",
    "current_bill", "arrears", "payable_within_due_date", "payable_after_due_date",
)
NUMERIC_FIELDS = (
    "sanctioned_load_kw", "previous_reading", "current_reading", "units_consumed",
    "total_charges", "total_taxes", "current_bill", "arrears",
    "payable_within_due_date", "payable_after_due_date",
)
DATE_FIELDS = ("reading_date", "issue_date", "due_date")

_PROVIDER_ALIASES = {
    "KE": "KE", "KESC": "KE", "K-ELECTRIC": "KE", "K ELECTRIC": "KE", "KELECTRIC": "KE",
    "LESCO": "LESCO", "LAHORE ELECTRIC SUPPLY COMPANY": "LESCO",
    "IESCO": "IESCO", "ISLAMABAD ELECTRIC SUPPLY COMPANY": "IESCO",
}
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

_CNIC_RE = re.compile(r"\b\d{5}-?\d{7}-?\d\b")
_LONG_DIGITS_RE = re.compile(r"(?<![\d.,])\d{10,}(?![\d.,])")


def empty_record() -> dict[str, Any]:
    """A schema-valid record with every field present and null/empty."""
    rec: dict[str, Any] = {k: None for k in LEVEL1_KEYS}
    rec["charges"] = []
    rec["taxes"] = []
    return rec


# --------------------------------------------------------------------------- numbers
def to_number(value: Any) -> int | float | None:
    """Parse a printed amount. '38209CR' -> -38209, 'Rs. 1,234.50' -> 1234.5, junk -> None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            return None
        return _tidy(Decimal(str(value)))
    if not isinstance(value, str):
        return None
    s = value.strip().replace("\u2212", "-").replace("\u2013", "-")
    if not s or s.lower() in {"null", "none", "n/a", "na", "nil", "-", "--"}:
        return None
    negative = False
    if re.search(r"\bcr\b|cr$", s, re.I):
        negative = True
    if s.startswith("(") and s.endswith(")"):
        negative, s = True, s[1:-1]
    s = re.sub(r"(?i)\b(rs|pkr)\b\.?", "", s)
    s = re.sub(r"(?i)(cr|dr)\s*$", "", s.strip())
    s = s.replace(",", "").replace(" ", "")
    if s.startswith("-"):
        negative, s = True, s[1:]
    elif s.startswith("+"):
        s = s[1:]
    if not re.fullmatch(r"\d+(\.\d+)?|\.\d+", s):
        return None
    num = Decimal(s)
    return _tidy(-num if negative else num)


def _tidy(d: Decimal) -> int | float:
    try:
        return int(d) if d == d.to_integral_value() else float(d)
    except (InvalidOperation, ValueError):
        return 0


# --------------------------------------------------------------------------- dates
def to_date(value: Any) -> str | None:
    """Return YYYY-MM-DD or None. Numeric dates are read day-first (Pakistani bills)."""
    if not isinstance(value, str):
        return None
    s = value.strip()
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        return _mk_date(int(m[1]), int(m[2]), int(m[3]))
    m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?[\s\-/.,]*([A-Za-z]{3,9})[\s\-/.,]*(\d{2,4})", s)
    if m and m[2][:3].lower() in _MONTHS:
        return _mk_date(_year(m[3]), _MONTHS[m[2][:3].lower()], int(m[1]))
    m = re.fullmatch(r"(\d{1,2})[\-/.](\d{1,2})[\-/.](\d{2,4})", s)
    if m:
        return _mk_date(_year(m[3]), int(m[2]), int(m[1]))
    return None


def to_month(value: Any) -> str | None:
    """Return YYYY-MM or None ('MAR 2026', 'Jun-26', '2026-03-15' all accepted)."""
    if not isinstance(value, str):
        return None
    s = value.strip()
    m = re.fullmatch(r"(\d{4})-(\d{1,2})(?:-\d{1,2})?", s)
    if m and 1 <= int(m[2]) <= 12:
        return f"{int(m[1]):04d}-{int(m[2]):02d}"
    m = re.fullmatch(r"([A-Za-z]{3,9})[\s\-/.,]*(\d{2,4})", s)
    if m and m[1][:3].lower() in _MONTHS:
        return f"{_year(m[2]):04d}-{_MONTHS[m[1][:3].lower()]:02d}"
    d = to_date(s)
    return d[:7] if d else None


def _year(txt: str) -> int:
    y = int(txt)
    return 2000 + y if y < 100 else y


def _mk_date(y: int, mo: int, d: int) -> str | None:
    try:
        return date(y, mo, d).isoformat()
    except ValueError:
        return None


# --------------------------------------------------------------------------- normalise
def normalize_level1(raw: Any) -> tuple[dict[str, Any], list[str]]:
    """Coerce a model-produced dict into the exact schema. Returns (record, notes)."""
    notes: list[str] = []
    rec = empty_record()
    if not isinstance(raw, dict):
        return rec, ["model output was not a JSON object"]

    prov = raw.get("provider")
    if isinstance(prov, str):
        rec["provider"] = _PROVIDER_ALIASES.get(prov.strip().upper())
        if rec["provider"] is None and prov.strip():
            notes.append(f"unrecognised provider {prov!r} set to null")
    tariff = raw.get("tariff")
    if isinstance(tariff, str) and tariff.strip():
        rec["tariff"] = tariff.strip()

    for k in NUMERIC_FIELDS:
        v = raw.get(k)
        n = to_number(v)
        if v not in (None, "") and n is None:
            notes.append(f"{k}: unparseable value {v!r} set to null")
        rec[k] = n

    rec["bill_month"] = to_month(raw.get("bill_month"))
    if raw.get("bill_month") not in (None, "") and rec["bill_month"] is None:
        notes.append(f"bill_month: bad value {raw.get('bill_month')!r} set to null")
    for k in DATE_FIELDS:
        rec[k] = to_date(raw.get(k))
        if raw.get(k) not in (None, "") and rec[k] is None:
            notes.append(f"{k}: bad value {raw.get(k)!r} set to null")

    rec["charges"] = _norm_items(raw.get("charges"), CHARGE_TYPES, "other", "charges", notes)
    rec["taxes"] = _norm_items(raw.get("taxes"), TAX_TYPES, "other_tax", "taxes", notes)
    return rec, notes


def _norm_items(items: Any, allowed: tuple[str, ...], default: str, name: str,
                notes: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        amt = to_number(it.get("amount"))
        if amt is None or amt == 0:          # blank / zero-valued lines are skipped
            continue
        typ = str(it.get("type", "")).strip().lower()
        if typ not in allowed:
            notes.append(f"{name}: type {typ!r} not allowed, using {default!r}")
            typ = default
        if typ == "subsidy" and amt > 0:     # subsidies are recorded as deductions
            amt = -amt
        out.append({"type": typ, "amount": amt})
    return out


# --------------------------------------------------------------------------- validation
def scrub_pii(text: str) -> str:
    """Remove CNIC-shaped values and very long digit runs (account/meter style numbers)."""
    text = _CNIC_RE.sub("[redacted]", text)
    return _LONG_DIGITS_RE.sub("[redacted]", text)


def _walk_strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v)


def validate_level1(rec: Any) -> list[str]:
    """Structural problems only (empty list == valid). Nulls are always acceptable."""
    issues: list[str] = []
    if not isinstance(rec, dict):
        return ["record is not an object"]
    for k in LEVEL1_KEYS:
        if k not in rec:
            issues.append(f"missing key {k}")
    extra = set(rec) - set(LEVEL1_KEYS)
    if extra:
        issues.append(f"unexpected keys {sorted(extra)}")
    if rec.get("provider") not in (None, *PROVIDERS):
        issues.append("provider not in KE/LESCO/IESCO")
    for k in NUMERIC_FIELDS:
        v = rec.get(k)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))
                              or isinstance(v, float) and not math.isfinite(v)):
            issues.append(f"{k} is not a number")
    if rec.get("bill_month") is not None and to_month(str(rec["bill_month"])) != rec["bill_month"]:
        issues.append("bill_month format")
    for k in DATE_FIELDS:
        if rec.get(k) is not None and to_date(str(rec[k])) != rec[k]:
            issues.append(f"{k} format")
    for name, allowed in (("charges", CHARGE_TYPES), ("taxes", TAX_TYPES)):
        items = rec.get(name)
        if not isinstance(items, list):
            issues.append(f"{name} is not a list")
            continue
        for it in items:
            if not isinstance(it, dict) or it.get("type") not in allowed \
                    or isinstance(it.get("amount"), bool) or not isinstance(it.get("amount"), (int, float)) \
                    or isinstance(it.get("amount"), float) and not math.isfinite(it["amount"]):
                issues.append(f"bad {name} entry {it!r}")
    for s in _walk_strings(rec):
        if _CNIC_RE.search(s) or _LONG_DIGITS_RE.search(s):
            issues.append("possible personal identifier in output")
            break
    try:
        json.loads(json.dumps(rec))
    except (TypeError, ValueError):
        issues.append("not JSON serialisable")
    return issues


def consistency_issues(rec: dict[str, Any], tol: float = 1.0) -> list[str]:
    """Arithmetic cross-checks against *printed* totals. These flag likely misreads;
    they never modify values."""
    out: list[str] = []
    tc, tt, cb = rec.get("total_charges"), rec.get("total_taxes"), rec.get("current_bill")
    charges = [c["amount"] for c in rec.get("charges") or []]
    taxes = [t["amount"] for t in rec.get("taxes") or []]
    if tc is not None and len(charges) >= 3 and abs(sum(charges) - tc) > tol:
        out.append(f"charge lines sum to {round(sum(charges), 2)} but total_charges is {tc}")
    if tt is not None and taxes and abs(sum(taxes) - tt) > tol:
        out.append(f"tax lines sum to {round(sum(taxes), 2)} but total_taxes is {tt}")
    if None not in (tc, tt, cb) and abs(tc + tt - cb) > 2 * tol:
        out.append(f"total_charges + total_taxes = {round(tc + tt, 2)} but current_bill is {cb}")
    pw, pa = rec.get("payable_within_due_date"), rec.get("payable_after_due_date")
    if None not in (pw, pa) and pa < pw:
        out.append(f"payable_after_due_date ({pa}) is lower than payable_within_due_date ({pw})")
    return out


def dumps_one_line(rec: dict[str, Any]) -> str:
    return json.dumps(rec, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
