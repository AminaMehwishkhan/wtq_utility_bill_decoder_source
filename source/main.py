"""Utility Bill Decoder - CLI.

  python main.py check-api                       # one tiny call to confirm key + model work
  python main.py run --bills-dir test/bills \\
        --level1-template test/csv/level1.csv --level2-template test/csv/level2.csv --out-dir output
  python main.py audit --out-dir output --level1-template ... --level2-template ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

from dotenv import load_dotenv

import csv_writer as cw
from bill_processor import BillError, BillProcessor, find_bill_file
from llm_client import LLMClient, LLMError, load_config
from schemas import dumps_one_line, empty_record, validate_level1

INSUFFICIENT = "This could not be answered because the bill could not be processed ({reason})."


def _client() -> LLMClient:
    load_dotenv()
    return LLMClient(load_config())


def cmd_check_api(_args) -> int:
    try:
        client = _client()
        text = client.generate("Reply with JSON only.", 'Return exactly {"ok": true}', max_tokens=64)
    except LLMError as e:
        print(f"API check FAILED: {e}")
        print("Tip: confirm API_KEY/MODEL_PROVIDER/MODEL_NAME in .env; a permitted alternative is "
              "MODEL_NAME=gemini-3.1-flash-lite (or set FALLBACK_MODEL_NAME).")
        return 1
    print(f"API check OK: provider={client.cfg.provider} model={client.model} reply={text.strip()[:60]!r}")
    return 0


def cmd_run(args) -> int:
    t0 = time.time()
    bills_dir, out_dir = Path(args.bills_dir), Path(args.out_dir)
    if not bills_dir.is_dir():
        print(f"error: bills directory not found: {bills_dir}")
        return 2
    try:
        client = _client()
    except LLMError as e:
        print(f"error: {e}")
        return 2

    # --- inputs (templates are the source of truth for ids, order and questions)
    if args.level1_template:
        l1_fields, l1_rows = cw.read_level1_template(args.level1_template)
    else:
        ids = sorted(p.stem for p in bills_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        l1_fields, l1_rows = cw.L1_COLS, [{"bill_id": i, "json": ""} for i in ids]
    l2_fields, l2_rows = (cw.read_level2_template(args.level2_template) if args.level2_template
                          else (cw.L2_COLS, []))
    if args.only:
        l1_rows = [r for r in l1_rows if r["bill_id"] in args.only]
        l2_rows = [r for r in l2_rows if r["bill_id"] in args.only]
    for r in l2_rows:
        r.setdefault("answer", "")
    print(f"{len(l1_rows)} bills, {len(l2_rows)} questions; model={client.cfg.provider}/{client.model}")

    proc = BillProcessor(client, args.cache_dir, use_cache=not args.fresh, repair=not args.no_repair)
    records, answers, report = OrderedDict(), {}, {"bills": {}, "model": client.cfg.model}
    questions_by_bill: dict[str, list[dict[str, str]]] = {}
    for r in l2_rows:
        questions_by_bill.setdefault(r["bill_id"], []).append(
            {"question_id": r["question_id"], "question": r["question"]})

    for row in l1_rows:
        bid = row["bill_id"]
        info = report["bills"][bid] = {"level1": "ok", "warnings": [], "level2_fallbacks": 0}
        print(f"[{bid}]")
        evidence = path = None
        try:
            path = find_bill_file(bills_dir, bid)
            evidence = proc.extract(bid, path)
            records[bid] = evidence["level1"]
            info["warnings"] = evidence["warnings"]
        except (BillError, LLMError, OSError) as e:
            records[bid] = empty_record()
            info["level1"] = f"FAILED: {e}"
            print(f"  level 1 failed: {e}")
        qs = questions_by_bill.get(bid, [])
        if not qs:
            continue
        if evidence is None:
            for q in qs:
                answers[(bid, q["question_id"])] = INSUFFICIENT.format(reason=info["level1"])
            info["level2_fallbacks"] = len(qs)
            continue
        try:
            got = proc.answer(evidence, qs, image_path=path)
        except (BillError, LLMError, OSError) as e:
            got = {q["question_id"]: INSUFFICIENT.format(reason=str(e)) for q in qs}
        for q in qs:
            ans = got.get(q["question_id"]) or INSUFFICIENT.format(reason="no answer produced")
            answers[(bid, q["question_id"])] = ans
            if ans.startswith("This question could not be answered") or ans.startswith("This could not"):
                info["level2_fallbacks"] += 1

    # --- outputs
    cw.write_level1(out_dir / "level1.csv", l1_fields, l1_rows, records)
    if l2_rows:
        cw.write_level2(out_dir / "level2.csv", l2_fields, l2_rows, answers)
    (out_dir / "run_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    # --- completeness report
    failed = [b for b, i in report["bills"].items() if i["level1"] != "ok"]
    fb = sum(i["level2_fallbacks"] for i in report["bills"].values())
    print(f"\nDone in {time.time() - t0:.0f}s | model calls: {client.calls}")
    print(f"Level 1: {len(records) - len(failed)}/{len(records)} bills extracted" +
          (f" (FAILED: {', '.join(failed)})" if failed else ""))
    if l2_rows:
        print(f"Level 2: {len(answers) - fb}/{len(answers)} answered normally, {fb} fallback statements")
    for b, i in report["bills"].items():
        for w in i["warnings"]:
            print(f"  note {b}: {w}")
    l1p, l1w = cw.audit_level1(out_dir / "level1.csv", args.level1_template)
    l2p, l2w = ([], [])
    if l2_rows:
        l2p, l2w = cw.audit_level2(out_dir / "level2.csv", args.level2_template)
    for p in l1p + l2p:
        print(f"  AUDIT PROBLEM: {p}")
    return 1 if (l1p or l2p or failed) else 0


def cmd_audit(args) -> int:
    out = Path(args.out_dir)
    l1p, l1w = cw.audit_level1(out / "level1.csv", args.level1_template)
    l2p, l2w = cw.audit_level2(out / "level2.csv", args.level2_template)
    for tag, items in (("PROBLEM", l1p + l2p), ("warning", l1w + l2w)):
        for i in items:
            print(f"{tag}: {i}")
    print("AUDIT PASSED" if not (l1p or l2p) else f"AUDIT FAILED ({len(l1p + l2p)} problem(s))")
    return 1 if (l1p or l2p) else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Utility Bill Decoder (WTQ 2026)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-api", help="verify API key and model with one tiny request")
    r = sub.add_parser("run", help="process a folder of bills and write level1.csv / level2.csv")
    r.add_argument("--bills-dir", required=True)
    r.add_argument("--level1-template")
    r.add_argument("--level2-template")
    r.add_argument("--out-dir", default="output")
    r.add_argument("--cache-dir", default=".cache")
    r.add_argument("--fresh", action="store_true", help="ignore cached results")
    r.add_argument("--no-repair", action="store_true", help="skip the re-check call for suspicious values")
    r.add_argument("--only", nargs="*", help="process only these bill_ids")
    a = sub.add_parser("audit", help="validate generated CSVs against the templates")
    a.add_argument("--out-dir", default="output")
    a.add_argument("--level1-template")
    a.add_argument("--level2-template")
    args = ap.parse_args(argv)
    return {"check-api": cmd_check_api, "run": cmd_run, "audit": cmd_audit}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
