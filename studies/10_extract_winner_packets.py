"""Phase 2 step 6: extract focused review packets for headline-winner identification.

For each paper, write a single text file containing:
  - Title + venue + year + citations
  - Abstract (full)
  - Conclusion section (full or first ~3000 chars)
  - Last 2000 chars (often contains the strongest summary statements)
  - Any sentence containing 'best', 'outperform', 'lowest loss', 'highest r-squared',
    'wins', 'beats', 'superior', 'dominant'

This produces studies/per_paper/<paper_id>/winner_packet.txt for human review.
"""
from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PER_PAPER = ROOT / "per_paper"


def pdftotext(pdf: Path) -> str:
    try:
        r = subprocess.run(
            ["pdftotext", str(pdf), "-"],
            capture_output=True, text=True, timeout=60,
        )
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def extract_abstract(text: str) -> str:
    lines = text.splitlines()
    start = end = None
    for i, ln in enumerate(lines):
        s = ln.strip().lower()
        if start is None and re.match(r"^\s*abstract\s*$|^\s*abstract[\s:.]", s):
            start = i + 1 if re.match(r"^\s*abstract\s*$", s) else i
            continue
        if start is not None and end is None:
            if re.match(r"^\s*(?:1\.?\s+)?introduction\s*$", s):
                end = i
                break
            if re.match(r"^\s*keywords?[\s:]", s):
                end = i
                break
            if re.match(r"^\s*jel\s+(?:codes?|classification)", s):
                end = i
                break
    if start is None:
        return text[:2000].strip()
    if end is None:
        end = min(start + 80, len(lines))
    s = " ".join(lines[start:end])
    return re.sub(r"\s+", " ", s).strip()[:3000]


def extract_conclusion(text: str) -> str:
    """Find a Conclusion / Concluding Remarks section."""
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        s = ln.strip().lower()
        if re.match(r"^\s*\d{0,2}\.?\s*(?:conclusion|concluding remarks|summary and conclusion)\s*$", s):
            start = i + 1
            break
        if re.match(r"^\s*(?:conclusion|concluding remarks)\s*$", s):
            start = i + 1
            break
    if start is None:
        return ""
    end = min(start + 200, len(lines))
    # Stop at References / Appendix
    for j in range(start, end):
        s2 = lines[j].strip().lower()
        if re.match(r"^\s*(?:references|appendix|acknowledg)", s2):
            end = j
            break
    s = " ".join(lines[start:end])
    return re.sub(r"\s+", " ", s).strip()[:5000]


def extract_winner_sentences(text: str) -> list[str]:
    """Sentences containing claim-of-winning language."""
    keywords = [
        r"\bbest\b", r"\boutperform[a-z]*\b", r"\blowest\b", r"\bhighest\b",
        r"\bsmallest\b", r"\bwins?\b", r"\bbeats?\b", r"\bsuperior\b",
        r"\bdominan[a-z]*\b", r"\boptimal\b", r"\btop[\- ]performing\b",
        r"\bmost accurate\b", r"\bachieves? the lowest\b",
    ]
    pat = "|".join(keywords)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    matches = []
    for s in sentences:
        if re.search(pat, s, flags=re.IGNORECASE):
            s_clean = re.sub(r"\s+", " ", s).strip()
            if 30 < len(s_clean) < 500:
                matches.append(s_clean)
    return matches[:30]


def main():
    p = ROOT / "candidate_30.csv"
    with p.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        pid = r["paper_id"]
        pdf = PER_PAPER / pid / "source.pdf"
        if not pdf.exists():
            continue
        text = pdftotext(pdf)
        if not text:
            continue
        abstract = extract_abstract(text)
        conclusion = extract_conclusion(text)
        winner_sents = extract_winner_sentences(text)
        last_2k = re.sub(r"\s+", " ", text[-2500:])

        out = []
        out.append(f"=== {pid} ===")
        out.append(f"Title:    {r['title']}")
        out.append(f"Venue:    {r['venue']}")
        out.append(f"Year:     {r['year']}, Citations: {r['citations']}")
        out.append("")
        out.append("--- ABSTRACT ---")
        out.append(abstract)
        out.append("")
        out.append("--- CONCLUSION (first 5000 chars) ---")
        out.append(conclusion if conclusion else "(no labeled conclusion section found)")
        out.append("")
        out.append("--- WINNER-CLAIM SENTENCES (top 30) ---")
        for s in winner_sents:
            out.append(f"- {s}")
        out.append("")
        out.append("--- LAST 2500 CHARS (often summary/conclusion) ---")
        out.append(last_2k)
        out.append("")

        outpath = pdf.parent / "winner_packet.txt"
        outpath.write_text("\n".join(out))
        print(f"  {pid}: packet written ({len(winner_sents)} winner sentences)")


if __name__ == "__main__":
    sys.exit(main() or 0)
