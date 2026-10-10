"""Safe CSV reading/writing and the final output audit (always via the csv module)."""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from schemas import dumps_one_line, scrub_pii, validate_level1

L1_COLS = ["bill_id", "json"]
L2_COLS = ["bill_id", "question_id", "question", "answer"]


def _read(path: str | Path, required: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"CSV not found: {p}")
    with p.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        names = list(reader.fieldnames or [])
        missing = [c for c in required if c not in names]
        if missing:
            raise ValueError(f"{p.name}: missing column(s) {missing}; found {names}")
        return names, [dict(r) for r in reader]


def read_level1_template(path) -> tuple[list[str], list[dict[str, str]]]:
    return _read(path, ["bill_id"])


def read_level2_template(path) -> tuple[list[str], list[dict[str, str]]]:
    return _read(path, ["bill_id", "question_id", "question"])


def _atomic_write(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_level1(path, fieldnames: list[str], template_rows: list[dict[str, str]],
                 records: dict[str, dict[str, Any]]) -> None:
    """One row per template row, original order and ids; JSON on a single line."""
    out = []
    for row in template_rows:
        r = dict(row)
        r["json"] = dumps_one_line(records[row["bill_id"]])
        out.append(r)
    _atomic_write(Path(path), fieldnames, out)


def write_level2(path, fieldnames: list[str], template_rows: list[dict[str, str]],
                 answers: dict[tuple[str, str], str]) -> None:
    """Copy bill_id/question_id/question untouched; only `answer` is written."""
    out = []
    for row in template_rows:
        r = dict(row)
        r["answer"] = answers[(row["bill_id"], row["question_id"])]
        out.append(r)
    _atomic_write(Path(path), fieldnames, out)


# ----------------------------------------------------------------------------- audit
def audit_level1(out_path, template_path=None) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    warnings: list[str] = []
    try:
        names, rows = _read(out_path, L1_COLS)
    except (FileNotFoundError, ValueError) as e:
        return [str(e)], warnings
    if names != L1_COLS:
        problems.append(f"level1 header is {names}, expected {L1_COLS}")
    if template_path:
        _, trows = read_level1_template(template_path)
        if [r["bill_id"] for r in rows] != [r["bill_id"] for r in trows]:
            problems.append("level1 bill_ids/order differ from the template")
    for r in rows:
        bid = r["bill_id"]
        raw = (r.get("json") or "").strip()
        if not raw:
            problems.append(f"{bid}: empty json cell")
            continue
        if "\n" in raw:
            problems.append(f"{bid}: JSON spans multiple lines")
        try:
            rec = json.loads(raw)
        except ValueError as e:
            problems.append(f"{bid}: JSON does not parse ({e})")
            continue
        problems += [f"{bid}: {i}" for i in validate_level1(rec)]
        nulls = [k for k, v in rec.items() if v is None]
        if len(nulls) >= 8:
            warnings.append(f"{bid}: {len(nulls)} null fields (extraction may have failed)")
    return problems, warnings


def audit_level2(out_path, template_path=None) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    warnings: list[str] = []
    try:
        names, rows = _read(out_path, L2_COLS)
    except (FileNotFoundError, ValueError) as e:
        return [str(e)], warnings
    if names != L2_COLS:
        problems.append(f"level2 header is {names}, expected {L2_COLS}")
    if template_path:
        _, trows = read_level2_template(template_path)
        if len(rows) != len(trows):
            problems.append(f"level2 has {len(rows)} rows, template has {len(trows)}")
        for i, (a, t) in enumerate(zip(rows, trows), start=2):
            for c in ("bill_id", "question_id", "question"):
                if a.get(c) != t.get(c):
                    problems.append(f"line {i}: {c} was changed")
    for r in rows:
        tag = f"{r['bill_id']}/{r['question_id']}"
        ans = (r.get("answer") or "").strip()
        if not ans:
            problems.append(f"{tag}: empty answer")
        elif scrub_pii(ans) != ans:
            problems.append(f"{tag}: answer contains a possible personal identifier")
        elif ans.startswith("This question could not be answered"):
            warnings.append(f"{tag}: fallback (insufficiency) answer")
    return problems, warnings
