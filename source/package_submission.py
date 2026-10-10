"""Build and verify submission.zip (layout from the participant guide):

  output/level1.csv, output/level2.csv, source/..., demo/demo.(mp4|mov|webm) [optional]

  python package_submission.py --output-dir output --level1-template test/csv/level1.csv \\
         --level2-template test/csv/level2.csv [--demo demo.mp4]

Refuses to package if the audit fails, a secret-looking string is found, or the ZIP exceeds 15 MB.
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

import csv_writer as cw

MAX_BYTES = 15 * 1000 * 1000
EXCLUDE_DIRS = {
    ".venv", "venv", "__pycache__", ".cache", "train", "test", "bills", "web_runs", ".git", "demo",
}
EXCLUDE_NAMES = {".env"}
EXCLUDE_SUFFIX = {".pyc", ".zip"}
KEY_PATTERNS = [re.compile(p) for p in (          # key-shaped strings: checked in every file
    r"AIza[0-9A-Za-z_\-]{30,}", r"sk-ant-[A-Za-z0-9_\-]{10,}", r"sk-[A-Za-z0-9_\-]{20,}")]
ENV_ASSIGN = re.compile(r"(?im)^\s*API_KEY\s*=\s*(?!your-api-key-here\s*$)\S+")   # only in .env* files
REQUIRED = ["output/level1.csv", "output/level2.csv", "source/README.md", "source/.env.example",
            "source/main.py", "source/requirements.txt"]


def source_files(src: Path) -> list[Path]:
    out = []
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if not p.is_file() or p.name in EXCLUDE_NAMES or p.suffix in EXCLUDE_SUFFIX:
            continue
        if any(part in EXCLUDE_DIRS or part == "output" or part.startswith(("output_", ".cache_"))
               for part in rel.parts[:-1]):
            continue
        out.append(p)
    return out


def scan_secrets(files: list[Path]) -> list[str]:
    hits = []
    for p in files:
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pat in KEY_PATTERNS:
            if pat.search(text):
                hits.append(f"{p.name}: matches a key-shaped pattern")
        if p.name.startswith(".env") and ENV_ASSIGN.search(text):
            hits.append(f"{p.name}: API_KEY is set to a real-looking value")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", default=".")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--level1-template")
    ap.add_argument("--level2-template")
    ap.add_argument("--demo")
    ap.add_argument("--zip-path", default="submission.zip")
    a = ap.parse_args()
    src, out = Path(a.source_dir).resolve(), Path(a.output_dir)

    problems: list[str] = []
    l1p, l1w = cw.audit_level1(out / "level1.csv", a.level1_template)
    l2p, l2w = cw.audit_level2(out / "level2.csv", a.level2_template)
    problems += l1p + l2p
    for w in l1w + l2w:
        print(f"warning: {w}")
    for path, n, name in ((out / "level1.csv", 10, "level1"), (out / "level2.csv", 120, "level2")):
        if path.is_file():
            rows = len(cw._read(path, [])[1])
            if rows != n:
                print(f"warning: {name}.csv has {rows} rows (the test set should have {n})")

    files = source_files(src)
    problems += [f"possible secret in {h}" for h in scan_secrets(files)]
    if (src / ".env").exists():
        print("note: .env exists locally and is excluded from the ZIP")
    demo = Path(a.demo) if a.demo else None
    if demo and (not demo.is_file() or demo.suffix.lower() not in (".mp4", ".mov", ".webm")):
        problems.append(f"demo file missing or not mp4/mov/webm: {demo}")
    if problems:
        print("NOT PACKAGED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    zp = Path(a.zip_path)
    if zp.exists():
        zp.unlink()
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out / "level1.csv", "output/level1.csv")
        z.write(out / "level2.csv", "output/level2.csv")
        for p in files:
            z.write(p, "source/" + p.relative_to(src).as_posix())
        if demo:
            z.write(demo, f"demo/demo{demo.suffix.lower()}")
    size = zp.stat().st_size
    with zipfile.ZipFile(zp) as z:
        bad = z.testzip()
        names = z.namelist()
    missing = [r for r in REQUIRED if r not in names]
    if bad or missing or size > MAX_BYTES:
        print(f"PACKAGE INVALID: corrupt={bad} missing={missing} size={size}")
        return 1
    print(f"OK: {zp} ({size / 1e6:.2f} MB, limit 15 MB), {len(names)} files")
    for n in names:
        print("  " + n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
