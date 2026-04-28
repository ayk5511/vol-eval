"""Phase 2 step 1: bibliographic search for candidate papers.

Queries Crossref + OpenAlex + arXiv for volatility-forecasting comparison
papers (2018-2025) in the target venues defined in PHASE_2_AND_3.md and the
companion paper's section 5.3.

No API keys required. Uses public APIs with polite User-Agent + email.

Outputs (in studies/results/):
- candidates_raw.csv:      all hits across the three sources, deduped by DOI/title
- candidates_filtered.csv: subset that passes the horse-race language filter

Run:  python3 studies/04_search_candidates.py
"""
from __future__ import annotations

import csv
import json
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

USER_AGENT = "vol-eval-survey/0.1 (mailto:1819ak@gmail.com)"

# Target journals per companion paper section 5.3
TARGET_JOURNALS = {
    "Journal of Financial Econometrics": "1479-8409",
    "Journal of Empirical Finance": "0927-5398",
    "Quantitative Finance": "1469-7688",
    "Journal of Forecasting": "1099-131X",
    "International Journal of Forecasting": "0169-2070",
    "Journal of Financial Data Science": "2640-3943",
}

YEAR_FROM = "2018-01-01"
YEAR_UNTIL = "2025-12-31"

# Filter keywords applied to title + abstract (lowercase substring match)
VOL_TERMS = ("volatility",)
FORECAST_TERMS = ("forecast", "predict")
COMPARISON_TERMS = (
    "comparison", "compare", "horse race", "horse-race", "versus", " vs ", " vs.",
    "out-of-sample", "out of sample", "model selection", "model confidence",
    "evaluation", "diebold", "mariano", "spa", "reality check", "benchmark",
    "competing", "outperform", "performance",
)
EXCLUDE_TITLE_TERMS = (
    "survey", "review", "overview", "literature review", "meta-analysis",
)


def http_get(url: str, timeout: int = 60, accept: str = "application/json") -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": accept}
    )
    with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as r:
        return r.read()


def crossref_search(issn: str, journal_name: str, max_per_journal: int = 400) -> list[dict]:
    rows: list[dict] = []
    cursor = "*"
    while len(rows) < max_per_journal:
        params = {
            "query.bibliographic": "volatility forecasting",
            "filter": (
                f"issn:{issn},"
                f"from-pub-date:{YEAR_FROM},"
                f"until-pub-date:{YEAR_UNTIL},"
                "type:journal-article"
            ),
            "rows": "100",
            "cursor": cursor,
            "select": "DOI,title,container-title,published-print,published-online,abstract,is-referenced-by-count,author",
        }
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
        try:
            payload = json.loads(http_get(url))
        except Exception as e:
            print(f"  crossref error {journal_name}: {e}", file=sys.stderr)
            break
        items = payload.get("message", {}).get("items", [])
        if not items:
            break
        for it in items:
            rows.append(_normalize_crossref(it, journal_name))
        cursor = payload.get("message", {}).get("next-cursor")
        if not cursor or len(items) < 100:
            break
        time.sleep(0.2)
    return rows


def _normalize_crossref(item: dict, journal_name: str) -> dict:
    title = (item.get("title") or [""])[0]
    abstract = item.get("abstract") or ""
    abstract = re.sub(r"<[^>]+>", " ", abstract)
    abstract = re.sub(r"\s+", " ", abstract).strip()
    pub = item.get("published-print") or item.get("published-online") or {}
    parts = pub.get("date-parts", [[None]])[0] if pub else [None]
    year = parts[0] if parts else None
    authors = item.get("author") or []
    author_str = "; ".join(
        f"{a.get('family', '')}, {a.get('given', '')}".strip(", ").strip()
        for a in authors
    )
    return {
        "source": "crossref",
        "doi": item.get("DOI", ""),
        "title": title,
        "abstract": abstract,
        "venue": journal_name,
        "year": year,
        "citations": item.get("is-referenced-by-count", 0),
        "authors": author_str,
        "url": f"https://doi.org/{item.get('DOI', '')}" if item.get("DOI") else "",
    }


def openalex_search(issn: str, journal_name: str, max_per_journal: int = 400) -> list[dict]:
    rows: list[dict] = []
    cursor = "*"
    while len(rows) < max_per_journal:
        params = {
            "search": "volatility forecasting",
            "filter": (
                f"primary_location.source.issn:{issn},"
                f"from_publication_date:{YEAR_FROM},"
                f"to_publication_date:{YEAR_UNTIL},"
                "type:article"
            ),
            "per_page": "100",
            "cursor": cursor,
            "mailto": "1819ak@gmail.com",
        }
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
        try:
            payload = json.loads(http_get(url))
        except Exception as e:
            print(f"  openalex error {journal_name}: {e}", file=sys.stderr)
            break
        results = payload.get("results", [])
        if not results:
            break
        for it in results:
            rows.append(_normalize_openalex(it, journal_name))
        cursor = payload.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(0.2)
    return rows


def _normalize_openalex(item: dict, journal_name: str) -> dict:
    abstract_idx = item.get("abstract_inverted_index") or {}
    abstract = _reconstruct_abstract(abstract_idx)
    authors = item.get("authorships") or []
    author_str = "; ".join(
        a.get("author", {}).get("display_name", "") for a in authors
    )
    doi = (item.get("doi") or "").replace("https://doi.org/", "")
    return {
        "source": "openalex",
        "doi": doi,
        "title": item.get("title", "") or "",
        "abstract": abstract,
        "venue": journal_name,
        "year": item.get("publication_year"),
        "citations": item.get("cited_by_count", 0),
        "authors": author_str,
        "url": item.get("doi") or item.get("id", ""),
    }


def _reconstruct_abstract(inverted: dict) -> str:
    if not inverted:
        return ""
    pos_to_word: dict[int, str] = {}
    for word, positions in inverted.items():
        for p in positions:
            pos_to_word[p] = word
    return " ".join(pos_to_word[k] for k in sorted(pos_to_word))


def arxiv_search(max_results: int = 600) -> list[dict]:
    """arXiv q-fin volatility-forecasting preprints, paginated."""
    rows: list[dict] = []
    start = 0
    page_size = 100
    while start < max_results:
        params = {
            "search_query": (
                "(cat:q-fin.ST OR cat:q-fin.RM OR cat:q-fin.CP) "
                "AND abs:volatility AND (abs:forecast OR abs:forecasting OR abs:prediction)"
            ),
            "start": str(start),
            "max_results": str(page_size),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
        try:
            xml_bytes = http_get(url, timeout=120, accept="application/atom+xml")
        except Exception as e:
            print(f"  arxiv error: {e}", file=sys.stderr)
            break
        ns = {"a": "http://www.w3.org/2005/Atom"}
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as e:
            print(f"  arxiv parse error: {e}", file=sys.stderr)
            break
        entries = root.findall("a:entry", ns)
        if not entries:
            break
        for e in entries:
            row = _normalize_arxiv(e, ns)
            if row and row["year"] and 2018 <= row["year"] <= 2025:
                rows.append(row)
        if len(entries) < page_size:
            break
        start += page_size
        time.sleep(3.5)  # arXiv asks for ≥3s between requests
    return rows


def _normalize_arxiv(entry, ns) -> dict | None:
    arxiv_id = entry.findtext("a:id", default="", namespaces=ns).split("/")[-1]
    title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
    title = re.sub(r"\s+", " ", title)
    summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
    summary = re.sub(r"\s+", " ", summary)
    published = entry.findtext("a:published", default="", namespaces=ns) or ""
    year = int(published[:4]) if published[:4].isdigit() else None
    authors = [
        a.findtext("a:name", default="", namespaces=ns)
        for a in entry.findall("a:author", ns)
    ]
    return {
        "source": "arxiv",
        "doi": "",
        "title": title,
        "abstract": summary,
        "venue": "arXiv q-fin",
        "year": year,
        "citations": 0,
        "authors": "; ".join(authors),
        "url": f"https://arxiv.org/abs/{arxiv_id}",
    }


def passes_filter(row: dict) -> tuple[bool, str]:
    text = (row.get("title", "") + " " + row.get("abstract", "")).lower()
    if not any(t in text for t in VOL_TERMS):
        return False, "no_volatility_term"
    if not any(t in text for t in FORECAST_TERMS):
        return False, "no_forecast_term"
    if not any(t in text for t in COMPARISON_TERMS):
        return False, "no_comparison_term"
    title_lower = row.get("title", "").lower()
    if any(t in title_lower for t in EXCLUDE_TITLE_TERMS):
        return False, "excluded_term_in_title"
    return True, "ok"


def dedupe(rows: list[dict]) -> list[dict]:
    """Dedupe by DOI when present, else by normalized title."""
    seen_doi: set[str] = set()
    seen_title: set[str] = set()
    out: list[dict] = []
    # Sort so crossref entries (with citation counts) come first
    rows = sorted(rows, key=lambda r: (r["source"] != "crossref", r["source"] != "openalex"))
    for r in rows:
        doi_key = r.get("doi", "").lower().strip()
        title_key = re.sub(r"\W+", "", (r.get("title") or "").lower())[:80]
        if doi_key and doi_key in seen_doi:
            continue
        if not doi_key and title_key and title_key in seen_title:
            continue
        if doi_key:
            seen_doi.add(doi_key)
        if title_key:
            seen_title.add(title_key)
        out.append(r)
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "source", "doi", "title", "abstract", "venue", "year",
        "citations", "authors", "url", "filter_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    print(f"Phase 2 candidate search: {YEAR_FROM} to {YEAR_UNTIL}")
    all_rows: list[dict] = []

    print("\n=== Crossref ===")
    for jname, issn in TARGET_JOURNALS.items():
        rows = crossref_search(issn, jname)
        print(f"  {jname}: {len(rows)} hits")
        all_rows.extend(rows)

    print("\n=== OpenAlex ===")
    for jname, issn in TARGET_JOURNALS.items():
        rows = openalex_search(issn, jname)
        print(f"  {jname}: {len(rows)} hits")
        all_rows.extend(rows)

    print("\n=== arXiv q-fin (newest 600) ===")
    rows = arxiv_search()
    print(f"  arXiv: {len(rows)} hits in 2018-2025")
    all_rows.extend(rows)

    print(f"\nTotal raw hits: {len(all_rows)}")
    deduped = dedupe(all_rows)
    print(f"After dedupe: {len(deduped)}")

    for r in deduped:
        passed, reason = passes_filter(r)
        r["filter_status"] = "pass" if passed else f"drop:{reason}"

    raw_path = RESULTS / "candidates_raw.csv"
    filtered_path = RESULTS / "candidates_filtered.csv"
    write_csv(deduped, raw_path)
    write_csv([r for r in deduped if r["filter_status"] == "pass"], filtered_path)

    n_pass = sum(1 for r in deduped if r["filter_status"] == "pass")
    print(f"\nWrote {raw_path}  ({len(deduped)} rows)")
    print(f"Wrote {filtered_path}  ({n_pass} rows)")

    by_source = Counter(r["source"] for r in deduped if r["filter_status"] == "pass")
    by_venue = Counter(r["venue"] for r in deduped if r["filter_status"] == "pass")
    print("\nFiltered set by source:")
    for s, c in by_source.most_common():
        print(f"  {s}: {c}")
    print("\nFiltered set by venue:")
    for v, c in by_venue.most_common():
        print(f"  {v}: {c}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
