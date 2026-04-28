"""Phase 2 setup audit: verify everything we have so far is consistent.

Checks:
1. candidate_30.csv well-formed: 30 unique paper_ids, valid years, etc.
2. candidate_30_access.csv synced with candidate_30.csv
3. per_paper/<paper_id>/source.pdf exists and is a valid PDF, for all 30
4. PDF first-page text matches the expected title (catches mis-mapped files)
5. Sample frame: which §5.3 venues are present/absent
6. Companion paper §5.3 venue list vs. actual sample

Run: python3 studies/08_audit_phase2_setup.py
"""
from __future__ import annotations

import csv
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PER_PAPER = ROOT / "per_paper"

CHECKS_PASS: list[str] = []
CHECKS_FAIL: list[str] = []
WARNINGS: list[str] = []


def ok(msg: str) -> None:
    CHECKS_PASS.append(msg)
    print(f"  PASS  {msg}")


def fail(msg: str) -> None:
    CHECKS_FAIL.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  WARN  {msg}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_candidate_30() -> list[dict]:
    section("candidate_30.csv well-formed")
    p = ROOT / "candidate_30.csv"
    if not p.exists():
        fail(f"{p} missing")
        return []
    with p.open() as f:
        rows = list(csv.DictReader(f))

    if len(rows) == 30:
        ok("30 rows present")
    else:
        fail(f"expected 30 rows, got {len(rows)}")

    pids = [r["paper_id"] for r in rows]
    if len(set(pids)) == len(pids):
        ok("paper_ids unique")
    else:
        dupes = [pid for pid, n in Counter(pids).items() if n > 1]
        fail(f"duplicate paper_ids: {dupes}")

    # Each row needs either a doi or an arxiv_id
    for r in rows:
        if not (r["doi"] or r["arxiv_id"]):
            fail(f"{r['paper_id']}: no DOI and no arxiv_id")

    dois = [r["doi"] for r in rows if r["doi"]]
    if len(set(dois)) == len(dois):
        ok(f"DOIs unique ({len(dois)} non-empty)")
    else:
        fail("duplicate DOIs")

    arx = [r["arxiv_id"] for r in rows if r["arxiv_id"]]
    if len(set(arx)) == len(arx):
        ok(f"arxiv_ids unique ({len(arx)} non-empty)")
    else:
        fail("duplicate arxiv_ids")

    for r in rows:
        try:
            y = int(r["year"])
            if not (2018 <= y <= 2026):
                warn(f"{r['paper_id']}: year={y} outside 2018-2026")
        except ValueError:
            fail(f"{r['paper_id']}: invalid year='{r['year']}'")

    return rows


def check_access_csv(rows_canon: list[dict]) -> None:
    section("candidate_30_access.csv synced")
    p = ROOT / "candidate_30_access.csv"
    if not p.exists():
        fail(f"{p} missing")
        return
    with p.open() as f:
        rows = list(csv.DictReader(f))
    canon_pids = {r["paper_id"] for r in rows_canon}
    access_pids = {r["paper_id"] for r in rows}
    if canon_pids == access_pids:
        ok("paper_ids in access CSV match candidate_30")
    else:
        missing = canon_pids - access_pids
        extra = access_pids - canon_pids
        if missing:
            fail(f"in candidate_30 but not access: {sorted(missing)}")
        if extra:
            fail(f"in access but not candidate_30: {sorted(extra)}")


def check_pdfs(rows_canon: list[dict]) -> dict[str, Path]:
    section("per_paper/<paper_id>/source.pdf integrity")
    found: dict[str, Path] = {}
    for r in rows_canon:
        pid = r["paper_id"]
        pdf = PER_PAPER / pid / "source.pdf"
        if not pdf.exists():
            fail(f"{pid}: source.pdf missing")
            continue
        if pdf.stat().st_size < 50_000:
            warn(f"{pid}: source.pdf only {pdf.stat().st_size // 1024}KB")
        with pdf.open("rb") as f:
            head = f.read(8)
        if head[:4] != b"%PDF":
            fail(f"{pid}: source.pdf is not a PDF (header={head!r})")
            continue
        found[pid] = pdf
    if len(found) == len(rows_canon):
        ok(f"all {len(found)} PDFs present and valid")
    return found


def pdf_first_page_text(pdf: Path, max_chars: int = 1500) -> str:
    """Extract first-page text via pdftotext (mac via brew or built-in)."""
    try:
        r = subprocess.run(
            ["pdftotext", "-l", "1", "-layout", str(pdf), "-"],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode == 0:
            return r.stdout[:max_chars]
    except FileNotFoundError:
        pass
    return ""


def check_pdf_titles(rows_canon: list[dict], found: dict[str, Path]) -> None:
    section("PDF first-page text matches expected title (catches mismaps)")
    if not found:
        return
    # Check pdftotext availability
    test = pdf_first_page_text(next(iter(found.values())), max_chars=200)
    if not test:
        warn("pdftotext not available - skipping title-vs-PDF cross-check")
        warn("install with: brew install poppler  (provides pdftotext)")
        return

    n_match = 0
    n_partial = 0
    n_mismatch = 0
    for r in rows_canon:
        pid = r["paper_id"]
        if pid not in found:
            continue
        expected = re.sub(r"\W+", "", r["title"].lower())
        text = pdf_first_page_text(found[pid])
        text_norm = re.sub(r"\W+", "", text.lower())
        # Look for any 30-char substring of expected title in the PDF text
        if not expected:
            continue
        # Use first 5 distinctive words of title
        title_words = [w for w in re.findall(r"\w+", r["title"].lower()) if len(w) > 3][:5]
        text_words = re.findall(r"\w+", text.lower())
        text_word_set = set(text_words)
        hits = sum(1 for w in title_words if w in text_word_set)
        if hits >= 3:
            n_match += 1
        elif hits >= 1:
            n_partial += 1
            warn(f"{pid}: title-match weak ({hits}/{len(title_words)} title-words found)")
        else:
            n_mismatch += 1
            fail(f"{pid}: title-match FAILED ({hits}/{len(title_words)} title-words found in first page)")
    if n_mismatch == 0:
        ok(f"all PDFs have title-words matching ({n_match} strong, {n_partial} weak)")


def check_sample_frame(rows_canon: list[dict]) -> None:
    section("sample frame vs. companion paper §5.3 stated venues")
    stated = {
        "Journal of Financial Econometrics",
        "Journal of Empirical Finance",
        "Quantitative Finance",
        "Journal of Forecasting",
        "International Journal of Forecasting",
        "Journal of Financial Data Science",
        "SSRN/arXiv",  # paper says "SSRN working papers" - we treat arXiv as equivalent
    }
    by_venue: dict[str, int] = defaultdict(int)
    for r in rows_canon:
        v = r["venue"]
        by_venue[v] += 1

    print("\n  Actual venue mix:")
    for v, n in sorted(by_venue.items(), key=lambda x: -x[1]):
        print(f"    {n:>2}  {v}")

    # Match against stated
    actual_venues = set(by_venue.keys())
    actual_normalized = set()
    for v in actual_venues:
        if "arXiv" in v:
            actual_normalized.add("SSRN/arXiv")
        else:
            actual_normalized.add(v)

    in_stated = actual_normalized & stated
    not_in_stated = actual_normalized - stated
    stated_missing = stated - actual_normalized

    print(f"\n  Stated venues present in sample: {len(in_stated)}")
    for v in sorted(in_stated):
        print(f"    + {v}")
    if not_in_stated:
        warn(f"{len(not_in_stated)} venues in sample were NOT in §5.3:")
        for v in sorted(not_in_stated):
            warn(f"    - {v}  (need to update §5.3)")
    if stated_missing:
        warn(f"{len(stated_missing)} venues stated in §5.3 but ABSENT from sample:")
        for v in sorted(stated_missing):
            warn(f"    - {v}  (need to update §5.3 OR add papers)")


def check_pipeline_logic() -> None:
    section("pipeline scripts and downstream consumers")
    expected = [
        "04_search_candidates.py",
        "05_narrow_candidates.py",
        "06_check_unpaywall.py",
        "07_download_papers.py",
        "03_apply_voleval_phase2.py",
        "candidate_30.csv",
        "candidate_30_access.csv",
        "02_paper_collection_template.csv",
    ]
    for f in expected:
        if (ROOT / f).exists():
            ok(f"{f} present")
        else:
            fail(f"{f} missing")


def main() -> int:
    print("Phase 2 setup audit\n")
    rows = check_candidate_30()
    if not rows:
        return 1
    check_access_csv(rows)
    found = check_pdfs(rows)
    check_pdf_titles(rows, found)
    check_sample_frame(rows)
    check_pipeline_logic()

    print("\n" + "=" * 60)
    print(f"PASS:    {len(CHECKS_PASS)}")
    print(f"WARN:    {len(WARNINGS)}")
    print(f"FAIL:    {len(CHECKS_FAIL)}")
    if CHECKS_FAIL:
        print("\nFAILURES:")
        for f in CHECKS_FAIL:
            print(f"  - {f}")
        return 1
    print("\nALL CHECKS PASSED  (warnings above are advisory)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
