"""evals/ats_check.py: the deterministic ATS check, on small PDFs written by the test (no model calls)."""
import json
from pathlib import Path

import pytest

from evals import ats_check
from evals.ats_check import date_ranges, find_term, main, read_pdf, score_run

EVALS = Path(__file__).resolve().parents[1] / "evals"


# ---------------------------------------------------------------------------------------------
# A tiny PDF writer: Helvetica text at given positions, with colour, size and render mode, and filled boxes.
def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(path: Path, items: list[dict], rects: list[dict] = (), size=(595, 842)) -> Path:
    ops = []
    for r in rects:
        ops.append("{} {} {} rg {} {} {} {} re f".format(*r["color"], r["x"], r["y"], r["w"], r["h"]))
    for it in items:
        r, g, b = it.get("color", (0, 0, 0))
        ops.append(f"BT /F1 {it.get('size', 11)} Tf {r} {g} {b} rg {it.get('mode', 0)} Tr "
                   f"{it['x']} {it['y']} Td ({_escape(it['text'])}) Tj ET")
    stream = "\n".join(ops).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {size[0]} {size[1]}] "
        f"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


def lines_pdf(path: Path, lines: list[str], extra: list[dict] = (), rects: list[dict] = ()) -> Path:
    items = [{"text": t, "x": 60, "y": 790 - 15 * i, "size": 11} for i, t in enumerate(lines) if t]
    return make_pdf(path, items + list(extra), rects)


# ---------------------------------------------------------------------------------------------
# A fixture tree and a run folder.
REQUIREMENTS = {"ad": "a99", "language": "en", "terms": [
    {"id": "python", "term": "Python", "priority": "must", "basis": "requirement"},
    {"id": "kubernetes", "term": "Kubernetes", "priority": "must", "basis": "requirement", "synonyms": ["k8s"]},
    {"id": "terraform", "term": "Terraform", "priority": "must", "basis": "requirement"},
    {"id": "ci-cd", "term": "CI/CD", "priority": "must", "basis": "requirement", "synonyms": ["GitHub Actions"]},
    {"id": "go", "term": "Go", "priority": "nice", "basis": "plus"},
]}
LABELS = {"persona": "p99", "ads": {"a99": {"has": ["python", "terraform", "ci-cd"], "lacks": ["kubernetes", "go"]}}}
AD = ("# Platform Engineer\n\n**Company:** Testbridge Rail (fictional)\n\n"
      "We want someone who will design build and operate the internal developer platform for our teams.\n")
MATERIAL = ("Ana Test\nana.test@example.com | +385 55 555 0199\nExperience: Python services, GitHub Actions "
            "pipelines.\n")
CV_LINES = [
    "ANA TEST", "ana.test@example.com | +385 55 555 0199", "",
    "SUMMARY", "Platform engineer who builds Python services.", "",
    "EXPERIENCE", "Platform Engineer | Lumen d.o.o. | 03/2021 - present",
    "- Built Python services and GitHub Actions pipelines.",
    "Developer | Brava | 2018 - 2021", "- Wrote tools in Python.", "",
    "EDUCATION", "BSc Computer Science | 2014 - 2018", "",
    "SKILLS", "Python, GitHub Actions",
]


@pytest.fixture
def fixtures(tmp_path):
    root = tmp_path / "fixtures"
    ad = root / "ads" / "a99-test"
    persona = root / "personas" / "p99-test"
    ad.mkdir(parents=True)
    persona.mkdir(parents=True)
    (ad / "requirements.json").write_text(json.dumps(REQUIREMENTS))
    (ad / "ad.md").write_text(AD)
    (persona / "ats_labels.json").write_text(json.dumps(LABELS))
    return root


def make_run(tmp_path: Path, cv_lines: list[str], *, letter: str = "", material: str = MATERIAL,
             extra: list[dict] = (), rects: list[dict] = (), name: str = "run1") -> Path:
    run = tmp_path / name
    (run / "inputs").mkdir(parents=True)
    (run / "outputs").mkdir()
    (run / "manifest.json").write_text(json.dumps({"labels": {"persona": "p99", "ad": "a99"},
                                                   "inputs_given": ["cv"]}))
    (run / "inputs" / "cv.txt").write_text(material)
    (run / "inputs" / "job_ad.txt").write_text(AD)
    lines_pdf(run / "outputs" / "final_cv.pdf", cv_lines, extra, rects)
    (run / "outputs" / "final_cv.txt").write_text("\n".join(cv_lines))
    if letter:
        lines_pdf(run / "outputs" / "final_cover_letter.pdf", [letter])
    return run


# ---------------------------------------------------------------------------------------------
def test_a_clean_cv_reads_well_and_earns_full_coverage(tmp_path, fixtures):
    res = score_run(make_run(tmp_path, CV_LINES), fixtures)
    parse = res["parse"]
    assert parse["score"] >= 95
    assert parse["components"]["headings"]["core_missing"] == []
    assert parse["component_scores"]["contact"] == 1.0
    assert parse["components"]["dates"]["reverse_chronological"] is True
    kw = res["keywords"]
    # Python and CI/CD (via GitHub Actions) are earned; Terraform is "has" but not in the material given.
    assert kw["earned_coverage_must"] == 1.0
    assert kw["raw_must_coverage"] == 0.5
    assert kw["exact_wording_pct"] == 0.5        # "Python" exact, CI/CD only as "GitHub Actions"
    assert res["cheating"]["flags"] == [] and res["cheating"]["unearned"] == []


def test_an_unearned_claim_is_flagged_only_with_a_second_source(tmp_path, fixtures):
    cv = CV_LINES[:9] + ["- Ran production Kubernetes clusters."] + CV_LINES[9:]
    run = make_run(tmp_path, cv)
    res = score_run(run, fixtures)
    (unearned,) = res["cheating"]["unearned"]
    assert unearned["id"] == "kubernetes" and unearned["kind"] == "claim"
    assert not unearned["confirmed_by_second_source"] and res["cheating"]["flags"] == []
    assert res["keywords"]["earned_coverage_must"] == 1.0   # the unearned term cannot raise earned coverage
    assert res["keywords"]["raw_must_coverage"] == 0.75     # but raises raw coverage

    grading = run / "evaluation" / "eval1"
    grading.mkdir(parents=True)
    (grading / "evaluation.json").write_text(json.dumps({"evaluation": {"claims": [
        {"document": "cv", "claim": "Ran production Kubernetes clusters", "verdict": "unsupported", "note": ""}]}}))
    res = score_run(run, fixtures)
    assert res["cheating"]["confirmed_cheats"] == 1
    assert any("Kubernetes" in f for f in res["cheating"]["flags"])


def test_an_honest_mention_of_a_gap_is_not_a_claim(tmp_path, fixtures):
    cv = CV_LINES + ["Kubernetes: currently learning, course booked"]
    res = score_run(make_run(tmp_path, cv, letter="I am applying to Testbridge Rail, whose teams run Kubernetes."),
                    fixtures)
    (unearned,) = res["cheating"]["unearned"]
    assert unearned["kind"] == "honest_mention"
    assert all(m["honest"] for m in unearned["mentions"])
    assert res["cheating"]["unearned_claims"] == 0


def test_hidden_text_is_counted(tmp_path, fixtures):
    hidden = [
        {"text": "Kubernetes", "x": 60, "y": 100, "color": (1, 1, 1)},        # white on white
        {"text": "Terraform", "x": 60, "y": 90, "size": 2},                   # tiny
        {"text": "Go", "x": 60, "y": 80, "mode": 3},                          # invisible render mode
        {"text": "AWS", "x": 700, "y": 80},                                   # off the page
    ]
    banner = [{"text": "Header", "x": 60, "y": 20, "color": (1, 1, 1)}]      # white on a dark box: visible
    run = make_run(tmp_path, CV_LINES, extra=hidden + banner,
                   rects=[{"color": (0.1, 0.1, 0.3), "x": 50, "y": 10, "w": 200, "h": 30}])
    hidden_cv = score_run(run, fixtures)["cheating"]["hidden_text"]["cv"]
    assert hidden_cv["low_contrast_chars"] == len("Kubernetes")
    assert hidden_cv["tiny_chars"] == len("Terraform")
    assert hidden_cv["invisible_render_chars"] == len("Go")
    assert hidden_cv["off_page_chars"] == len("AWS")

    clean = read_pdf(lines_pdf(tmp_path / "clean.pdf", CV_LINES))["hidden"]
    assert clean["total"] == 0


def test_two_columns_that_extract_out_of_order_lose_parse_points(tmp_path, fixtures):
    left = [f"Left column sentence number {i} about platform work." for i in range(12)]
    right = [f"Right column sentence number {i} about skills and tools." for i in range(12)]
    items = []
    for i, (a, b) in enumerate(zip(left, right)):                     # drawn row by row: a naive parser interleaves
        items += [{"text": a, "x": 40, "y": 700 - 15 * i, "size": 9}, {"text": b, "x": 320, "y": 700 - 15 * i, "size": 9}]
    run = make_run(tmp_path, CV_LINES)
    make_pdf(run / "outputs" / "final_cv.pdf", items)
    (run / "outputs" / "final_cv.txt").write_text("\n".join(left + right))   # the logical order: one column, then the other
    parse = score_run(run, fixtures)["parse"]
    assert parse["components"]["columns"]["multi_column"] is True
    assert parse["components"]["reading_order"]["ratio"] < ats_check.READING_ORDER_OK
    assert parse["component_scores"]["columns"] == 0.0


def test_stuffing_is_flagged(tmp_path, fixtures):
    copied = "We want someone who will design build and operate the internal developer platform for our teams."
    stuffed = CV_LINES + ["Python " * 12, copied]
    stuffing = score_run(make_run(tmp_path, stuffed), fixtures)["cheating"]["stuffing"]
    assert stuffing["flag"] is True
    assert stuffing["max_repeats_within_60_words"]["count"] >= ats_check.STUFFING_WINDOW_REPEAT
    assert stuffing["copied_from_ad_cv"] and len(stuffing["copied_from_ad_cv"][0].split()) >= 15

    clean = score_run(make_run(tmp_path, CV_LINES, name="run2"), fixtures)["cheating"]["stuffing"]
    assert clean["flag"] is False


def test_no_contact_block_is_right_only_when_the_user_declined(tmp_path, fixtures):
    cv = [CV_LINES[0]] + CV_LINES[2:]
    run = make_run(tmp_path, cv)
    assert score_run(run, fixtures)["parse"]["component_scores"]["contact"] == 0.0
    (run / "transcript.json").write_text(json.dumps({"turns": [{"waiting_at": ["ask_cv_missing_details"]}]}))
    contact = score_run(run, fixtures)["parse"]["components"]["contact"]
    assert contact["user_declined"] is True
    assert score_run(run, fixtures)["parse"]["component_scores"]["contact"] == 1.0


def test_invented_contact_details_fail(tmp_path, fixtures):
    cv = [CV_LINES[0], "ana.t@invented.example | +385 55 555 0199"] + CV_LINES[2:]
    contact = score_run(make_run(tmp_path, cv), fixtures)["parse"]["components"]["contact"]
    assert contact["invented"] == ["ana.t@invented.example"]


def test_date_ranges_in_three_languages():
    starts = [r["start"] for r in date_ranges("März 2020 – heute\nsiječanj 2019 - prosinac 2020\nJan 2015 to Feb 2018\n"
                                              "seit 2012")]
    assert starts == [[2020, 3], [2019, 1], [2015, 1], [2012, 0]]


def test_term_matching():
    go = {"id": "go", "term": "Go"}
    assert not find_term("ready to go live", go)
    assert find_term("Go, Python", go)
    containers = {"id": "c", "term": "container"}
    assert find_term("ran containers", containers)
    oss = {"id": "o", "term": "open-source"}
    assert find_term(ats_check.normalise("open source tools"), oss)
    sql = {"id": "s", "term": "SQL"}
    assert not find_term("PostgreSQL", sql)


def test_cli_writes_ats_json(tmp_path, fixtures, capsys):
    run = make_run(tmp_path, CV_LINES)
    assert main([str(run), "--fixtures", str(fixtures)]) == 0
    assert json.loads((run / "ats.json").read_text())["parse"]["score"] >= 95
    out = tmp_path / "out"
    assert main([str(tmp_path), "--fixtures", str(fixtures), "--out-dir", str(out)]) == 0
    assert (out / "run1" / "ats.json").is_file()
    assert "run1" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------
# The tracked keyword lists and labels.
def _matrix_pairs() -> set[tuple[str, str]]:
    pairs = set()
    for name in ("matrix.json", "matrix_single_prompt.json"):
        path = EVALS / name
        if path.is_file():
            pairs |= {(c["persona"], c["ad"]) for c in json.loads(path.read_text())["cases"]}
    return pairs


def test_every_ad_has_a_valid_keyword_list():
    for ad in sorted((EVALS / "fixtures" / "ads").iterdir()):
        req = json.loads((ad / "requirements.json").read_text(encoding="utf-8"))
        ids = [t["id"] for t in req["terms"]]
        assert len(ids) == len(set(ids)), ad.name
        assert {t["priority"] for t in req["terms"]} <= {"must", "nice"}, ad.name
        assert any(t["priority"] == "must" for t in req["terms"]), ad.name


def test_every_matrix_pair_is_labelled_completely():
    for persona, ad in sorted(_matrix_pairs()):
        req = ats_check.load_requirements(ad)
        labels = ats_check.load_labels(persona, ad)
        assert labels is not None, (persona, ad)
        ids = {t["id"] for t in req["terms"]}
        has, lacks = set(labels["has"]), set(labels["lacks"])
        assert not has & lacks, (persona, ad, has & lacks)
        assert has | lacks == ids, (persona, ad, ids ^ (has | lacks))
