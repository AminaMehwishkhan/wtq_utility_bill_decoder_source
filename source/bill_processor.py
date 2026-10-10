"""Image handling, local OCR, and the two model-backed tasks (extraction + Q&A).

Design: the model sees the ORIGINAL bill image (orientation-corrected, resized) plus OCR text
as extra evidence. Python normalises/validates everything the model returns, does all
arithmetic, and keeps a per-bill evidence record that is checkpointed to disk.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageOps

import prompts
from calculations import CalcError, build_env, derived_facts, render_answer
from llm_client import LLMClient, LLMError
from schemas import (consistency_issues, normalize_level1, scrub_pii, to_month, to_number,
                     validate_level1)

try:  # optional: deskew/contrast for OCR only
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = np = None
try:  # optional: local OCR
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None

SUPPORTED_EXT = (".png", ".jpg", ".jpeg")
PIPELINE_VERSION = "2"          # bump to invalidate caches when prompts/logic change


class BillError(RuntimeError):
    """A bill could not be processed (message is safe to log)."""


# ----------------------------------------------------------------------------- images
def find_bill_file(bills_dir: str | Path, bill_id: str) -> Path:
    base = Path(bills_dir)
    for ext in SUPPORTED_EXT:
        for cand in (base / f"{bill_id}{ext}", base / f"{bill_id}{ext.upper()}"):
            if cand.is_file():
                return cand
    raise BillError(f"no image found for {bill_id} in {base} (looked for {', '.join(SUPPORTED_EXT)})")


def load_image(path: str | Path) -> Image.Image:
    """Load PNG/JPG, apply EXIF rotation, return RGB. The file on disk is never modified."""
    p = Path(path)
    if p.suffix.lower() not in SUPPORTED_EXT:
        raise BillError(f"unsupported image format {p.suffix!r} (use png/jpg)")
    try:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA", "P"):
                rgba = im.convert("RGBA")
                bg = Image.new("RGB", rgba.size, "white")
                bg.paste(rgba, mask=rgba.split()[-1])
                return bg
            return im.convert("RGB")
    except (OSError, ValueError) as e:
        raise BillError(f"cannot read image {p.name}: {e}") from None


def to_jpeg_bytes(img: Image.Image, max_edge: int = 2200, min_edge: int = 1600,
                  quality: int = 92) -> bytes:
    """Image sent to the model: shrink very large photos, gently upscale tiny ones."""
    w, h = img.size
    long_edge = max(w, h)
    scale = 1.0
    if long_edge > max_edge:
        scale = max_edge / long_edge
    elif long_edge < min_edge:
        scale = min(min_edge / long_edge, 3.0)
    if abs(scale - 1.0) > 1e-3:
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def estimate_skew(gray: "np.ndarray") -> float:
    """Median angle (degrees) of long near-horizontal lines (table borders, text rules)."""
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=120,
                            minLineLength=max(gray.shape[1] // 4, 50), maxLineGap=20)
    if lines is None:
        return 0.0
    angles = [float(np.degrees(np.arctan2(y2 - y1, x2 - x1))) for x1, y1, x2, y2 in lines[:, 0]]
    angles = [a for a in angles if abs(a) <= 12]
    return float(np.median(angles)) if len(angles) >= 5 else 0.0


def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """Grey, upscale small text, deskew modest tilt (2-10 degrees is typical for photos).
    No binarisation or contrast stretching: measured on the training bills, upscale-only matched
    or beat upscale+CLAHE on every bill, and faint decimals/minus signs must survive."""
    gray_img = ImageOps.grayscale(img)
    if cv2 is None:
        return gray_img
    gray = np.array(gray_img)
    if gray.shape[1] < 1800:
        f = min(3.0, 1800 / gray.shape[1])
        gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    angle = estimate_skew(gray)
    if 0.3 <= abs(angle) <= 10:
        h, w = gray.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        gray = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(gray)


_ocr_state: dict[str, Any] = {"lang": None, "warned": False}


def _ocr_lang() -> str:
    if _ocr_state["lang"] is None:
        try:
            langs = set(pytesseract.get_languages(config=""))
        except Exception:
            langs = set()
        _ocr_state["lang"] = "eng+urd" if {"eng", "urd"} <= langs else "eng"
    return _ocr_state["lang"]


def run_ocr(img: Image.Image, log: Callable[[str], None] = print) -> str:
    """Local Tesseract OCR; returns '' (never raises) when unavailable. Urdu used if installed."""
    if pytesseract is None:
        return ""
    cmd = os.environ.get("TESSERACT_CMD", "").strip()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    elif not shutil.which("tesseract"):
        if not _ocr_state["warned"]:
            log("  [warn] Tesseract not found; continuing without OCR evidence (vision-only).")
            _ocr_state["warned"] = True
        return ""
    try:
        return pytesseract.image_to_string(preprocess_for_ocr(img), lang=_ocr_lang(),
                                           config="--psm 6", timeout=90)
    except Exception as e:  # missing language data, timeouts, binary errors
        if not _ocr_state["warned"]:
            log(f"  [warn] OCR failed ({type(e).__name__}); continuing without OCR evidence.")
            _ocr_state["warned"] = True
        return ""


# ----------------------------------------------------------------------------- JSON
def parse_json_loose(text: str) -> Any:
    """Parse model output: strips code fences, trims chatter, repairs trailing commas."""
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    s = s[start:end + 1]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", s))


# ----------------------------------------------------------------------------- processor
class BillProcessor:
    def __init__(self, client: LLMClient, cache_dir: str | Path = ".cache", use_cache: bool = True,
                 repair: bool = True, log: Callable[[str], None] = print):
        self.client, self.use_cache, self.repair, self.log = client, use_cache, repair, log
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- cache helpers
    def _read(self, name: str, key: str) -> dict | None:
        p = self.cache_dir / name
        if self.use_cache and p.is_file():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                return d if d.get("key") == key else None
            except (OSError, ValueError):
                return None
        return None

    def _write(self, name: str, payload: dict) -> None:
        tmp = self.cache_dir / (name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.cache_dir / name)

    def _ask_json(self, system: str, user: str, images: list[bytes], max_tokens: int = 12000) -> dict:
        """One model call that must yield a JSON object; one corrective retry on bad JSON."""
        text = self.client.generate(system, user, images, max_tokens)
        try:
            obj = parse_json_loose(text)
        except ValueError as e:
            text = self.client.generate(
                system, user + f"\n\nYour previous reply was not valid JSON ({e}). "
                "Reply again with ONLY the JSON object.", images, max_tokens)
            try:
                obj = parse_json_loose(text)
            except ValueError as e2:
                raise BillError(f"model returned invalid JSON twice ({e2})") from None
        if not isinstance(obj, dict):
            raise BillError("model JSON was not an object")
        return obj

    # -- Level 1
    def extract(self, bill_id: str, path: str | Path) -> dict[str, Any]:
        """Return the evidence record (level1 + history + lines + ocr) for one bill."""
        path = Path(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        key = f"{PIPELINE_VERSION}:{sha}:{self.client.cfg.provider}:{self.client.cfg.model}"
        cached = self._read(f"{bill_id}.extract.json", key)
        if cached:
            self.log(f"  {bill_id}: extraction loaded from cache")
            return cached["evidence"]

        img = load_image(path)
        jpeg = to_jpeg_bytes(img)
        ocr = run_ocr(img, self.log)
        raw = self._ask_json(prompts.EXTRACT_SYSTEM, prompts.extract_user(ocr), [jpeg])
        parsed = self._interpret(raw)
        warnings = list(parsed["warnings"])

        issues = self._issues(parsed["level1"])
        if issues and self.repair:
            self.log(f"  {bill_id}: re-checking {len(issues)} suspicious value(s)")
            try:
                raw2 = self._ask_json(prompts.EXTRACT_SYSTEM,
                                      prompts.repair_user(raw.get("level1", raw), issues), [jpeg])
                parsed2 = self._interpret(raw2)
                if len(self._issues(parsed2["level1"])) < len(issues):
                    parsed, warnings = parsed2, list(parsed2["warnings"]) + ["values corrected on re-check"]
                else:
                    warnings.append("re-check did not resolve: " + "; ".join(issues))
            except (BillError, LLMError) as e:
                warnings.append(f"re-check failed ({e}); kept first extraction")
        elif issues:
            warnings.append("unresolved: " + "; ".join(issues))

        evidence = {"bill_id": bill_id, "model": self.client.cfg.model, "ocr_text": scrub_pii(ocr),
                    "level1": parsed["level1"], "history": parsed["history"],
                    "bill_lines": parsed["bill_lines"], "other_values": parsed["other_values"],
                    "notes": parsed["notes"], "warnings": warnings}
        self._write(f"{bill_id}.extract.json", {"key": key, "evidence": evidence})
        return evidence

    @staticmethod
    def _issues(rec: dict[str, Any]) -> list[str]:
        out = consistency_issues(rec)
        if rec.get("provider") is None:
            out.append("provider could not be identified")
        if rec.get("current_bill") is None:
            out.append("current_bill is missing")
        return out

    @staticmethod
    def _interpret(raw: dict[str, Any]) -> dict[str, Any]:
        l1_raw = raw.get("level1") if isinstance(raw.get("level1"), dict) else raw
        rec, warnings = normalize_level1(l1_raw)
        problems = validate_level1(rec)
        if any("personal identifier" in p for p in problems):
            rec["tariff"] = None
            warnings.append("tariff looked like an identifier; set to null")
        bill_month = rec.get("bill_month")
        history = []
        for h in raw.get("history") or []:
            if not isinstance(h, dict):
                continue
            month = to_month(h.get("month"))
            if not month:
                continue
            history.append({
                "month": month, "units": to_number(h.get("units")),
                "billed_amount": to_number(h.get("billed_amount")),
                "payment": to_number(h.get("payment")),
                "status": h.get("status") if isinstance(h.get("status"), str) else None,
                "is_current_month": bool(h.get("is_current_month")) or month == bill_month})
        lines = []
        for ln in raw.get("bill_lines") or []:
            if isinstance(ln, dict) and ln.get("label"):
                lines.append({"section": str(ln.get("section") or "")[:60], "label": str(ln["label"])[:80],
                              "units": to_number(ln.get("units")), "rate": to_number(ln.get("rate")),
                              "amount": to_number(ln.get("amount"))})
        others = [{"label": str(o["label"])[:80], "value": scrub_pii(str(o.get("value", "")))[:120]}
                  for o in (raw.get("other_values") or [])[:40] if isinstance(o, dict) and o.get("label")]
        notes = scrub_pii(str(raw.get("notes") or ""))[:600]
        return {"level1": rec, "history": history, "bill_lines": lines, "other_values": others,
                "notes": notes, "warnings": warnings}

    # -- Level 2
    def answer(self, evidence: dict[str, Any], questions: list[dict[str, str]],
               image_path: str | Path | None = None) -> dict[str, str]:
        """questions: [{'question_id','question'}] -> {question_id: answer}. The bill image is
        re-attached (when image_path is given) so answers stay grounded in the bill itself.
        Never raises for one bad answer; failures become truthful 'could not answer' statements
        and are not cached, so a re-run retries them."""
        bill_id = evidence["bill_id"]
        key = f"{PIPELINE_VERSION}:{self.client.cfg.provider}:{self.client.cfg.model}:" + \
              hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
        stored = (self._read(f"{bill_id}.answers.json", key) or {}).get("answers", {})
        results: dict[str, str] = {}
        todo = []
        for q in questions:
            qh = hashlib.sha256(q["question"].encode()).hexdigest()
            hit = stored.get(q["question_id"])
            if hit and hit.get("q") == qh:
                results[q["question_id"]] = hit["answer"]
            else:
                todo.append(q)
        if not todo:
            self.log(f"  {bill_id}: answers loaded from cache")
            return results

        level1, history = evidence["level1"], evidence["history"]
        env = build_env(level1, history)
        facts = {"level1": level1, "history": history, "bill_lines": evidence["bill_lines"],
                 "other_values": evidence["other_values"], "extraction_notes": evidence["notes"],
                 "derived_ratios": derived_facts(level1)}
        jpeg = to_jpeg_bytes(load_image(image_path)) if image_path else None
        failures: dict[str, str] = {}
        gave_up: set[str] = set()
        got = self._ask_answers(facts, env, todo, evidence["ocr_text"], jpeg, failures)
        results.update(got)

        retry = [q for q in todo if q["question_id"] in failures]
        for q in retry:                                   # one bounded retry per failed question
            reason = failures[q["question_id"]]
            self.log(f"  {bill_id}: retrying {q['question_id']} ({reason})")
            f2: dict[str, str] = {}
            extra = dict(facts, previous_attempt_problem=reason + ". Use only listed VARIABLES or numbers from the "
                         "facts; if an expression cannot express it, compute carefully and write final numbers directly.")
            got2 = self._ask_answers(extra, env, [q], evidence["ocr_text"], jpeg, f2)
            if q["question_id"] in got2:
                results[q["question_id"]] = got2[q["question_id"]]
            else:
                gave_up.add(q["question_id"])
                results[q["question_id"]] = ("This question could not be answered reliably from the bill "
                                             f"data that was extracted ({f2.get(q['question_id'], reason)}).")
        cacheable = {q["question_id"]: {"q": hashlib.sha256(q["question"].encode()).hexdigest(),
                                        "answer": results[q["question_id"]]}
                     for q in questions
                     if q["question_id"] in results and q["question_id"] not in gave_up}
        self._write(f"{bill_id}.answers.json", {"key": key, "answers": cacheable})
        return results

    def _ask_answers(self, facts, env, questions, ocr, jpeg, failures: dict[str, str]) -> dict[str, str]:
        preview = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in env.items()}
        qs = [{"question_id": q["question_id"], "question": q["question"]} for q in questions]
        try:
            obj = self._ask_json(prompts.ANSWER_SYSTEM, prompts.answers_user(facts, preview, qs, ocr),
                                 [jpeg] if jpeg else [], max_tokens=8192)
        except (BillError, LLMError) as e:
            for q in questions:
                failures[q["question_id"]] = f"model call failed: {e}"
            return {}
        by_id = {}
        for a in obj.get("answers") or []:
            if isinstance(a, dict) and a.get("question_id") is not None:
                by_id[str(a["question_id"]).strip()] = a
        out: dict[str, str] = {}
        for q in questions:
            qid = q["question_id"]
            a = by_id.get(qid)
            if not a or not isinstance(a.get("answer"), str) or not a["answer"].strip():
                failures[qid] = "no usable answer returned"
                continue
            try:
                text = render_answer(a["answer"], a.get("calcs"), env)
            except CalcError as e:
                failures[qid] = f"calculation problem: {e}"
                continue
            out[qid] = scrub_pii(re.sub(r"\s+", " ", text).strip())
        return out
