"""Run:  TRAIN_BILLS_DIR=path/to/train/bills python -m unittest discover -s tests -v

IMPORTANT: no real model is called here. LLM behaviour is simulated with a scripted fake client,
so these tests prove the plumbing (parsing, normalisation, retries, CSV integrity, error handling),
NOT the model's accuracy. Real accuracy needs a live key: see README ("Validate on training bills").
OCR/preprocessing tests use the real training images when TRAIN_BILLS_DIR is set.
"""
import csv
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import calculations as calc  # noqa: E402
import csv_writer as cw  # noqa: E402
import llm_client as lc  # noqa: E402
import main as cli  # noqa: E402
import package_submission as package  # noqa: E402
import schemas as sc  # noqa: E402
from bill_processor import (BillError, BillProcessor, find_bill_file, load_image,  # noqa: E402
                            parse_json_loose, preprocess_for_ocr, run_ocr, to_jpeg_bytes)
from compare_reference import diff  # noqa: E402

TRAIN = Path(os.environ.get("TRAIN_BILLS_DIR", ROOT / "train" / "bills"))
OFFICIAL = json.loads((ROOT / "tests/reference/KESC_0008.json").read_text(encoding="utf-8"))
OFFICIAL.pop("_source")
HISTORY_0008 = [("2025-04", 141), ("2025-05", 138), ("2025-06", 197), ("2025-07", 239), ("2025-08", 203),
                ("2025-09", 176), ("2025-10", 173), ("2025-11", 187), ("2025-12", 124), ("2026-01", 96),
                ("2026-02", 88), ("2026-03", 115), ("2026-04", 151)]


def extraction_reply(level1=None, **extra):
    d = {"level1": level1 or OFFICIAL,
         "history": [{"month": m, "units": u, "billed_amount": None, "payment": None, "status": None,
                      "is_current_month": m == "2026-04"} for m, u in HISTORY_0008],
         "bill_lines": [{"section": "charges", "label": "Fixed Charges", "units": 3, "rate": 300, "amount": 900}],
         "other_values": [{"label": "Energy charges share", "value": "83.39%"}], "notes": ""}
    d.update(extra)
    return json.dumps(d)


class FakeClient:
    """Scripted stand-in for LLMClient: replies are popped in order (str or Exception)."""
    def __init__(self, replies):
        self.cfg = lc.LLMConfig(api_key="x")
        self.model = self.cfg.model
        self.replies, self.calls, self.prompts = list(replies), 0, []

    def generate(self, system, user, images=None, max_tokens=0):
        self.calls += 1
        self.prompts.append((system, user, len(images or [])))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Tmp(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)


# ------------------------------------------------------------------------ schemas
class TestSchemas(unittest.TestCase):
    def test_numbers_dates_and_credits(self):
        self.assertEqual(sc.to_number("38209CR"), -38209)
        self.assertEqual(sc.to_number("Rs. 19,964"), 19964)
        self.assertEqual(sc.to_number("-0.53"), -0.53)
        self.assertIsNone(sc.to_number("illegible"))
        self.assertEqual(sc.to_date("22nd Sep 2026"), "2026-09-22")
        self.assertEqual(sc.to_date("26 AUG 26"), "2026-08-26")
        self.assertIsNone(sc.to_date("31-Feb-26"))
        self.assertEqual(sc.to_month("MAR 2026"), "2026-03")

    def test_normalise_rules(self):
        rec, notes = sc.normalize_level1({
            "provider": "K-Electric", "total_taxes": "illegible", "current_bill": "3,430.24",
            "arrears": "54770CR", "due_date": "13th Jul 2026",
            "charges": [{"type": "energy", "amount": 10}, {"type": "energy", "amount": "20.5"},
                        {"type": "fixed", "amount": 0}, {"type": "subsidy", "amount": 3332},
                        {"type": "weird", "amount": 5}],
            "taxes": [{"type": "gst", "amount": 7}]})
        self.assertEqual(rec["provider"], "KE")
        self.assertIsNone(rec["total_taxes"])                       # unreadable -> null, not a guess
        self.assertEqual(rec["current_bill"], 3430.24)
        self.assertEqual(rec["arrears"], -54770)                    # negative credit kept
        self.assertEqual(rec["due_date"], "2026-07-13")
        self.assertEqual([c["type"] for c in rec["charges"]], ["energy", "energy", "subsidy", "other"])
        self.assertEqual(rec["charges"][2]["amount"], -3332)         # subsidy forced negative
        self.assertEqual(sc.validate_level1(rec), [])
        self.assertTrue(any("total_taxes" in n for n in notes))

    def test_validation_catches_problems(self):
        rec = sc.empty_record()
        self.assertEqual(sc.validate_level1(rec), [])
        rec["current_bill"] = "100"
        rec["provider"] = "WAPDA"
        rec["tariff"] = "12345-1234567-1"
        rec["bill_month"] = "2026-99"
        issues = " ".join(sc.validate_level1(rec))
        self.assertIn("current_bill", issues)
        self.assertIn("provider", issues)
        self.assertIn("bill_month", issues)
        self.assertIn("personal identifier", issues)
        rec = sc.empty_record()
        rec["current_bill"] = float("nan")
        rec["charges"] = [{"type": "energy", "amount": float("inf")}]
        issues = " ".join(sc.validate_level1(rec))
        self.assertIn("current_bill", issues)
        self.assertIn("bad charges entry", issues)
        with self.assertRaises(ValueError):
            sc.dumps_one_line(rec)

    def test_consistency_flags_without_changing(self):
        rec = sc.empty_record()
        rec.update(total_charges=100, total_taxes=20, current_bill=200,
                   charges=[{"type": "energy", "amount": 50}] * 3)
        self.assertEqual(len(sc.consistency_issues(rec)), 2)
        self.assertEqual(rec["current_bill"], 200)

    def test_official_example_is_internally_consistent(self):
        self.assertEqual(sc.consistency_issues(OFFICIAL), [])


# ------------------------------------------------------------------------ calculations
class TestCalculations(unittest.TestCase):
    def setUp(self):
        hist = [{"month": m, "units": u, "is_current_month": m == "2026-04"} for m, u in HISTORY_0008]
        self.env = calc.build_env(OFFICIAL, hist)

    def test_guide_sample_answers(self):
        self.assertAlmostEqual(calc.evaluate("pct(total_taxes, current_bill)", self.env), 16.6058, 3)
        self.assertEqual(calc.evaluate("count_gt(units_prev, 200)", self.env), 2)     # guide: two months
        self.assertEqual(calc.evaluate("count_gt(units_all, 200)", self.env), 2)
        self.assertEqual(calc.evaluate("len(units_prev)", self.env), 12)

    def test_render(self):
        txt = calc.render_answer("Taxes are {a} ({b}).", [
            {"id": "a", "expr": "pct(total_taxes,current_bill)", "format": "percent", "decimals": 1},
            {"id": "b", "expr": "total_taxes", "format": "pkr"}], self.env)
        self.assertEqual(txt, "Taxes are 16.6% (PKR 569.62).")
        txt = calc.render_answer("Taxes are PKR {b}, about {a}%.", [
            {"id": "a", "expr": "pct(total_taxes,current_bill)", "format": "percent", "decimals": 2},
            {"id": "b", "expr": "total_taxes", "format": "pkr"}], self.env)
        self.assertEqual(txt, "Taxes are PKR 569.62, about 16.61%.")
        txt = calc.render_answer("Taxes make up {b} PKR.", [
            {"id": "b", "expr": "total_taxes", "format": "pkr"}], self.env)
        self.assertEqual(txt, "Taxes make up PKR 569.62.")

    def test_rejects_unsafe_and_undefined(self):
        for bad in ("__import__('os').system('x')", "().__class__", "9**99", "1/0", "nope+1",
                    "max([])", "'a'*3", "open('f')", "units_prev.append(1)"):
            with self.assertRaises(calc.CalcError, msg=bad):
                calc.evaluate(bad, self.env)

    def test_unresolved_placeholder_and_null_variable(self):
        with self.assertRaises(calc.CalcError):
            calc.render_answer("Value {x}", [], self.env)
        with self.assertRaises(calc.CalcError):
            calc.evaluate("total_taxes", {"total_taxes": None})


# ------------------------------------------------------------------------ CSV
class TestCsv(Tmp):
    def _templates(self):
        l1, l2 = self.dir / "l1.csv", self.dir / "l2.csv"
        with l1.open("w", newline="", encoding="utf-8") as f:
            f.write("bill_id,json\nKESC_0008,\nLESCO_0005,\n")
        rows = [("KESC_0008", "Q1", 'How much is "tax", in total, پاکستان?'), ("KESC_0008", "Q2", "Line\nbreak?"),
                ("LESCO_0005", "Q1", "Plain?")]
        with l2.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cw.L2_COLS)
            for r in rows:
                w.writerow([*r, ""])
        return l1, l2

    def test_round_trip_and_audit(self):
        l1, l2 = self._templates()
        f1, r1 = cw.read_level1_template(l1)
        f2, r2 = cw.read_level2_template(l2)
        recs = {"KESC_0008": OFFICIAL, "LESCO_0005": sc.empty_record()}
        ans = {("KESC_0008", "Q1"): 'About 16.6%, i.e. PKR 569.62 of 3,430.24, "taxes", yes.',
               ("KESC_0008", "Q2"): "Two months,\nJuly and August.", ("LESCO_0005", "Q1"): "Not shown."}
        out = self.dir / "out"
        cw.write_level1(out / "level1.csv", f1, r1, recs)
        cw.write_level2(out / "level2.csv", f2, r2, ans)
        with (out / "level1.csv").open(newline="", encoding="utf-8") as f:
            back = list(csv.DictReader(f))
        self.assertEqual([r["bill_id"] for r in back], ["KESC_0008", "LESCO_0005"])
        self.assertEqual(json.loads(back[0]["json"]), OFFICIAL)
        self.assertNotIn("\n", back[0]["json"].strip())
        with (out / "level2.csv").open(newline="", encoding="utf-8") as f:
            back2 = list(csv.DictReader(f))
        self.assertEqual(back2[0]["question"], 'How much is "tax", in total, پاکستان?')
        self.assertEqual(back2[1]["answer"], ans[("KESC_0008", "Q2")])
        p1, _ = cw.audit_level1(out / "level1.csv", l1)
        p2, _ = cw.audit_level2(out / "level2.csv", l2)
        self.assertEqual((p1, p2), ([], []))

    def test_audit_detects_tampering(self):
        l1, l2 = self._templates()
        f2, r2 = cw.read_level2_template(l2)
        ans = {(r["bill_id"], r["question_id"]): "ok" for r in r2}
        cw.write_level2(self.dir / "o.csv", f2, r2, ans)
        text = (self.dir / "o.csv").read_text(encoding="utf-8").replace("Plain?", "Changed?")
        (self.dir / "o.csv").write_text(text, encoding="utf-8")
        problems, _ = cw.audit_level2(self.dir / "o.csv", l2)
        self.assertTrue(any("question was changed" in p for p in problems))
        ans[("LESCO_0005", "Q1")] = ""
        cw.write_level2(self.dir / "o2.csv", f2, r2, ans)
        problems, _ = cw.audit_level2(self.dir / "o2.csv", l2)
        self.assertTrue(any("empty answer" in p for p in problems))


# ------------------------------------------------------------------------ LLM client (HTTP mocked)
class Resp:
    def __init__(self, status=200, body=None, text="", headers=None):
        self.status_code, self._b, self.text, self.headers = status, body or {}, text, headers or {}

    def json(self):
        return self._b


GEMINI_OK = {"candidates": [{"content": {"parts": [{"text": "{\"ok\":true}"}]}, "finishReason": "STOP"}]}


def client(**kw):
    cfg = lc.LLMConfig(api_key="SECRET-KEY-123", min_interval=0, **kw)
    return lc.LLMClient(cfg, sleep=lambda s: None)


class TestLLMClient(unittest.TestCase):
    def test_retry_then_success(self):
        with mock.patch("llm_client.requests.post", side_effect=[Resp(429), Resp(503), Resp(200, GEMINI_OK)]) as post:
            self.assertEqual(client().generate("s", "u"), '{"ok":true}')
            self.assertEqual(post.call_count, 3)

    def test_gives_up_after_bounded_retries(self):
        with mock.patch("llm_client.requests.post", return_value=Resp(500, text="boom")) as post:
            with self.assertRaises(lc.LLMError):
                client(max_retries=2).generate("s", "u")
            self.assertEqual(post.call_count, 3)

    def test_auth_error_not_retried_and_key_redacted(self):
        with mock.patch("llm_client.requests.post", return_value=Resp(401, text="bad key SECRET-KEY-123")) as post:
            with self.assertRaises(lc.LLMError) as cm:
                client().generate("s", "u")
            self.assertEqual(post.call_count, 1)
            self.assertNotIn("SECRET-KEY-123", str(cm.exception))

    def test_model_not_found_uses_permitted_fallback_only(self):
        with mock.patch("llm_client.requests.post", side_effect=[Resp(404, text="nf"), Resp(200, GEMINI_OK)]) as post:
            c = client(fallback_model="gemini-3.1-flash-lite")
            c.generate("s", "u")
            self.assertIn("gemini-3.1-flash-lite", post.call_args_list[1].args[0])
        with self.assertRaises(lc.LLMError):
            client(fallback_model="gpt-4.1-mini")            # not permitted for google
        with mock.patch("llm_client.requests.post", return_value=Resp(404, text="nf")):
            with self.assertRaises(lc.LLMError):
                client().generate("s", "u")                   # no fallback configured

    def test_request_shapes_and_parsing(self):
        img = [b"\xff\xd8fake"]
        with mock.patch("llm_client.requests.post", return_value=Resp(200, GEMINI_OK)) as post:
            client().generate("sys", "usr", img)
            url, kw = post.call_args.args[0], post.call_args.kwargs
            self.assertIn("gemini-3.5-flash-lite:generateContent", url)
            self.assertEqual(kw["headers"]["x-goog-api-key"], "SECRET-KEY-123")
            self.assertNotIn("SECRET", url)
            self.assertIn("inlineData", json.dumps(kw["json"]))
        anth = {"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn"}
        with mock.patch("llm_client.requests.post", return_value=Resp(200, anth)) as post:
            c = lc.LLMClient(lc.LLMConfig(provider="anthropic", model="claude-haiku-4-5", api_key="k",
                                          min_interval=0), sleep=lambda s: None)
            self.assertEqual(c.generate("s", "u", img), "hi")
            self.assertEqual(post.call_args.kwargs["headers"]["anthropic-version"], "2023-06-01")
        oai = {"choices": [{"message": {"content": "yo"}, "finish_reason": "stop"}]}
        for model, key in (("gpt-5-mini", "max_completion_tokens"), ("gpt-4.1-mini", "max_tokens")):
            with mock.patch("llm_client.requests.post", return_value=Resp(200, oai)) as post:
                c = lc.LLMClient(lc.LLMConfig(provider="openai", model=model, api_key="k", min_interval=0),
                                 sleep=lambda s: None)
                self.assertEqual(c.generate("s", "u", img), "yo")
                self.assertIn(key, post.call_args.kwargs["json"])
                self.assertTrue(post.call_args.kwargs["headers"]["Authorization"].startswith("Bearer "))
class TestPackaging(Tmp):
    def test_source_files_exclude_local_inputs_and_outputs(self):
        (self.dir / "main.py").write_text("print('source')", encoding="utf-8")
        for folder in ("bills", "web_runs/run/output", "output_training", ".cache", ".cache_training", "test"):
            path = self.dir / folder
            path.mkdir(parents=True, exist_ok=True)
            (path / "private.txt").write_text("local data", encoding="utf-8")
        included = {path.relative_to(self.dir).as_posix() for path in package.source_files(self.dir)}
        self.assertEqual(included, {"main.py"})

    def test_truncated_and_empty_responses_raise(self):
        trunc = {"candidates": [{"content": {"parts": [{"text": "{\"a\":"}]}, "finishReason": "MAX_TOKENS"}]}
        with mock.patch("llm_client.requests.post", return_value=Resp(200, trunc)):
            with self.assertRaises(lc.LLMError):
                client().generate("s", "u")
        with mock.patch("llm_client.requests.post", return_value=Resp(200, {"candidates": []})):
            with self.assertRaises(lc.LLMError):
                client().generate("s", "u")


# ------------------------------------------------------------------------ pipeline with scripted model
class TestPipeline(Tmp):
    def setUp(self):
        super().setUp()
        self.img = TRAIN / "KESC_0008.png"
        if not self.img.exists():
            self.skipTest("training image KESC_0008.png not available (set TRAIN_BILLS_DIR)")
        self.bills = self.dir / "bills"
        self.bills.mkdir()
        shutil.copy(self.img, self.bills / "KESC_0008.png")

    def proc(self, replies, **kw):
        c = FakeClient(replies)
        return BillProcessor(c, self.dir / "cache", log=lambda *_: None, **kw), c

    def test_reference_plumbing_matches_official_json(self):
        p, c = self.proc([extraction_reply()])
        ev = p.extract("KESC_0008", self.bills / "KESC_0008.png")
        self.assertEqual(ev["level1"], OFFICIAL)
        self.assertEqual(diff(OFFICIAL, ev["level1"]), [])
        self.assertEqual(c.prompts[0][2], 1)                     # image was attached
        self.assertEqual(len(ev["history"]), 13)
        self.assertTrue(ev["history"][-1]["is_current_month"])

    def test_messy_model_output_is_repaired(self):
        messy = "Sure!\n```json\n" + extraction_reply(
            level1=dict(OFFICIAL, current_bill="3,430.24", arrears="-0.53", taxes=OFFICIAL["taxes"])).rstrip("}") + ",}\n```"
        p, c = self.proc([messy])
        ev = p.extract("KESC_0008", self.bills / "KESC_0008.png")
        self.assertEqual(ev["level1"]["current_bill"], 3430.24)

    def test_invalid_json_twice_raises_bill_error(self):
        p, _ = self.proc(["not json", "still not json"])
        with self.assertRaises(BillError):
            p.extract("KESC_0008", self.bills / "KESC_0008.png")

    def test_unreadable_field_becomes_null_and_bill_survives(self):
        bad = dict(OFFICIAL, total_taxes="smudged", due_date="??")
        p, _ = self.proc([extraction_reply(bad)] * 2)
        ev = p.extract("KESC_0008", self.bills / "KESC_0008.png")
        self.assertIsNone(ev["level1"]["total_taxes"])
        self.assertIsNone(ev["level1"]["due_date"])
        self.assertEqual(ev["level1"]["current_bill"], 3430.24)

    def test_repair_accepted_when_it_resolves_issues(self):
        wrong = dict(OFFICIAL, total_charges=2360.62)           # misread digit
        p, c = self.proc([extraction_reply(wrong), extraction_reply(OFFICIAL)])
        ev = p.extract("KESC_0008", self.bills / "KESC_0008.png")
        self.assertEqual(ev["level1"]["total_charges"], 2860.62)
        self.assertEqual(c.calls, 2)

    def test_repair_rejected_when_it_does_not_help(self):
        wrong = dict(OFFICIAL, total_charges=2360.62)
        p, _ = self.proc([extraction_reply(wrong), extraction_reply(wrong)])
        ev = p.extract("KESC_0008", self.bills / "KESC_0008.png")
        self.assertEqual(ev["level1"]["total_charges"], 2360.62)  # printed value kept, flagged, not "fixed"
        self.assertTrue(any("did not resolve" in w for w in ev["warnings"]))

    def test_cache_prevents_second_call(self):
        p, c = self.proc([extraction_reply()])
        p.extract("KESC_0008", self.bills / "KESC_0008.png")
        p.extract("KESC_0008", self.bills / "KESC_0008.png")
        self.assertEqual(c.calls, 1)

    def test_answers_use_python_calculations(self):
        p, c = self.proc([extraction_reply()])
        ev = p.extract("KESC_0008", self.bills / "KESC_0008.png")
        reply = json.dumps({"answers": [
            {"question_id": "Q1", "answer": "Taxes are PKR 569.62 of PKR 3,430.24, about {a}.",
             "calcs": [{"id": "a", "expr": "pct(total_taxes, current_bill)", "format": "percent", "decimals": 1}]},
            {"question_id": "Q2", "answer": "{n} of the 12 previous months exceeded 200 units.",
             "calcs": [{"id": "n", "expr": "count_gt(units_prev, 200)", "format": "int"}]}]})
        c.replies.append(reply)
        qs = [{"question_id": "Q1", "question": "tax share?"}, {"question_id": "Q2", "question": "months > 200?"}]
        out = p.answer(ev, qs, image_path=self.bills / "KESC_0008.png")
        self.assertIn("about 16.6%", out["Q1"])
        self.assertTrue(out["Q2"].startswith("2 of the 12"))
        self.assertEqual(c.prompts[-1][2], 1)                    # image re-attached for grounding

    def test_bad_calc_triggers_one_retry_then_truthful_fallback(self):
        p, c = self.proc([extraction_reply()])
        ev = p.extract("KESC_0008", self.bills / "KESC_0008.png")
        bad = json.dumps({"answers": [{"question_id": "Q1", "answer": "It is {x}.",
                                       "calcs": [{"id": "x", "expr": "does_not_exist * 2"}]}]})
        c.replies += [bad, bad]
        out = p.answer(ev, [{"question_id": "Q1", "question": "?"}])
        self.assertTrue(out["Q1"].startswith("This question could not be answered"))
        self.assertEqual(c.calls, 3)                              # 1 extraction + batch + one retry

    def test_pii_scrubbed_from_answers(self):
        p, c = self.proc([extraction_reply()])
        ev = p.extract("KESC_0008", self.bills / "KESC_0008.png")
        c.replies.append(json.dumps({"answers": [{"question_id": "Q1",
                         "answer": "CNIC 35202-1234567-1 and meter 123456789012 are hidden.", "calcs": []}]}))
        out = p.answer(ev, [{"question_id": "Q1", "question": "?"}])
        self.assertNotIn("1234567", out["Q1"])

    def test_cli_end_to_end_with_one_failing_bill(self):
        l1, l2 = self.dir / "l1.csv", self.dir / "l2.csv"
        l1.write_text("bill_id,json\nKESC_0008,\nIESCO_0099,\n", encoding="utf-8")
        with l2.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cw.L2_COLS)
            for b in ("KESC_0008", "IESCO_0099"):
                w.writerow([b, "Q1", "Tax share, in %?", ""])
        fake = FakeClient([extraction_reply(), json.dumps({"answers": [
            {"question_id": "Q1", "answer": "About {a}.", "calcs": [
                {"id": "a", "expr": "pct(total_taxes,current_bill)", "format": "percent", "decimals": 1}]}]})])
        out = self.dir / "out"
        with mock.patch.object(cli, "_client", return_value=fake):
            code = cli.main(["run", "--bills-dir", str(self.bills), "--level1-template", str(l1),
                             "--level2-template", str(l2), "--out-dir", str(out),
                             "--cache-dir", str(self.dir / "c2")])
        self.assertEqual(code, 1)                                # one bill failed -> non-zero, but outputs exist
        rows = read_rows(out / "level1.csv")
        self.assertEqual(json.loads(rows[0]["json"])["current_bill"], 3430.24)
        failed = json.loads(rows[1]["json"])
        self.assertEqual(sc.validate_level1(failed), [])
        self.assertTrue(all(v is None for k, v in failed.items() if k not in ("charges", "taxes")))
        a = read_rows(out / "level2.csv")
        self.assertIn("16.6%", a[0]["answer"])
        self.assertTrue(a[1]["answer"].startswith("This could not be answered"))
        p1, _ = cw.audit_level1(out / "level1.csv", l1)
        p2, _ = cw.audit_level2(out / "level2.csv", l2)
        self.assertEqual((p1, p2), ([], []))


# ------------------------------------------------------------------------ real images: preprocessing + OCR
class TestRealImages(unittest.TestCase):
    def setUp(self):
        if not TRAIN.exists():
            self.skipTest("TRAIN_BILLS_DIR not set / training bills not present")

    def test_all_training_images_load_and_encode(self):
        for name in ("IESCO_0004", "KESC_0004", "KESC_0008", "LESCO_0005", "LESCO_0006"):
            path = find_bill_file(TRAIN, name)
            img = load_image(path)
            jpg = to_jpeg_bytes(img)
            self.assertGreater(len(jpg), 10_000)
            self.assertLess(len(jpg), 4_000_000)
            self.assertEqual(preprocess_for_ocr(img).mode, "L")

    def test_ocr_reads_reference_bill_values(self):
        import shutil as sh
        if not sh.which("tesseract"):
            self.skipTest("tesseract not installed")
        text = run_ocr(load_image(find_bill_file(TRAIN, "KESC_0008")), log=lambda *_: None)
        for token in ("3,430.24", "2,860.62", "569.62", "3,716", "151", "10.5400"):
            self.assertIn(token, text)

    def test_ocr_degrades_gracefully_when_missing(self):
        with mock.patch("bill_processor.shutil.which", return_value=None), \
                mock.patch.dict(os.environ, {"TESSERACT_CMD": ""}):
            import bill_processor as bp
            bp._ocr_state["warned"] = False
            self.assertEqual(run_ocr(load_image(find_bill_file(TRAIN, "IESCO_0004")), log=lambda *_: None), "")

    def test_unsupported_and_missing_files(self):
        with self.assertRaises(BillError):
            find_bill_file(TRAIN, "NOPE_0000")
        with tempfile.NamedTemporaryFile(suffix=".gif") as f:
            with self.assertRaises(BillError):
                load_image(f.name)


class TestJsonParsing(unittest.TestCase):
    def test_variants(self):
        self.assertEqual(parse_json_loose('```json\n{"a": 1,}\n```'), {"a": 1})
        self.assertEqual(parse_json_loose('Here: {"a": [1,2,]} thanks'), {"a": [1, 2]})
        with self.assertRaises(ValueError):
            parse_json_loose("no json here")


if __name__ == "__main__":
    unittest.main()
