"""Compare pipeline output with tests/reference/*.json (numbers within +/-1, text exact).

  python tests/compare_reference.py --cache-dir .cache
  python tests/compare_reference.py --level1-csv output/level1.csv
Only KESC_0008 is an official answer; the others are the assistant's own reading of the images,
so a mismatch there means "look at the bill", not necessarily "the pipeline is wrong".
Keys starting with '_' and keys absent from a reference are ignored.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REF_DIR = Path(__file__).parent / "reference"


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def diff(ref: dict, got: dict) -> list[str]:
    out = []
    for k, rv in ref.items():
        if k.startswith("_"):
            continue
        gv = got.get(k)
        if k in ("charges", "taxes"):
            rl = [(i["type"], i["amount"]) for i in rv]
            gl = [(i["type"], i["amount"]) for i in (gv or [])]
            ok = len(rl) == len(gl) and all(a[0] == b[0] and abs(a[1] - b[1]) <= 1 for a, b in zip(rl, gl))
            if not ok:
                out.append(f"{k}: expected {rl} got {gl}")
        elif _num(rv):
            if not _num(gv) or abs(rv - gv) > 1:
                out.append(f"{k}: expected {rv} got {gv}")
        elif rv != gv:
            out.append(f"{k}: expected {rv!r} got {gv!r}")
    return out


def load_outputs(args) -> dict[str, dict]:
    if args.level1_csv:
        with open(args.level1_csv, newline="", encoding="utf-8-sig") as f:
            return {r["bill_id"]: json.loads(r["json"]) for r in csv.DictReader(f) if r["json"].strip()}
    res = {}
    for p in Path(args.cache_dir).glob("*.extract.json"):
        res[p.name.split(".")[0]] = json.loads(p.read_text(encoding="utf-8"))["evidence"]["level1"]
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=".cache")
    ap.add_argument("--level1-csv")
    args = ap.parse_args()
    outputs = load_outputs(args)
    bad = checked = 0
    for ref_path in sorted(REF_DIR.glob("*.json")):
        bid = ref_path.stem
        if bid not in outputs:
            print(f"{bid}: no pipeline output found (skipped)")
            continue
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        d = diff(ref, outputs[bid])
        checked += 1
        bad += bool(d)
        print(f"{bid} [{ref.get('_source', '')[:40]}]: " + ("MATCH" if not d else f"{len(d)} difference(s)"))
        for line in d:
            print("   -", line)
    print(f"\n{checked - bad}/{checked} references fully matched")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
