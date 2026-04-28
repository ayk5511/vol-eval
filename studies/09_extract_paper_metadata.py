"""Phase 2 step 5: extract structured metadata from each PDF.

For each paper in candidate_30.csv:
- Extract title, abstract, full-text via pdftotext
- Detect code URLs (GitHub / GitLab / Zenodo / CodeOcean)
- Detect data sources (Yahoo, FRED, Bloomberg, named public datasets)
- Score RDS (0/1/2) per Khan2026Survey rubric (best-effort, with confidence)
- Extract headline winner / runner-up from abstract
- Detect model families compared (GARCH, HAR, ML, ensemble)
- Detect asset class (equity, crypto, FX, commodity)
- For JFEcon papers, also pull section structure for the style guide

Outputs:
- studies/per_paper/<paper_id>/metadata.json (per-paper structured record)
- studies/results/phase2_per_paper_metadata.csv (aggregated table)
- studies/results/structure_notes_jfec.csv (6 JFEcon papers, structure detail)

The script is best-effort and flags low-confidence rows for human review.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PER_PAPER = ROOT / "per_paper"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# --- Patterns ------------------------------------------------------------

CODE_URL_PATTERNS = [
    r"https?://(?:www\.)?github\.com/[\w\-./]+",
    r"https?://(?:www\.)?gitlab\.com/[\w\-./]+",
    r"https?://(?:www\.)?bitbucket\.org/[\w\-./]+",
    r"https?://(?:www\.)?codeocean\.com/[\w\-./]+",
    r"https?://(?:www\.)?zenodo\.org/[\w\-./]+",
    r"https?://(?:www\.)?osf\.io/[\w\-./]+",
    r"https?://(?:www\.)?figshare\.com/[\w\-./]+",
    r"https?://(?:www\.)?dataverse\.[\w\-.]+/[\w\-./]+",
]

PUBLIC_DATA_KEYWORDS = [
    "yahoo finance", "yahoo! finance", "yahoofinance", "finance.yahoo.com",
    "fred", "federal reserve economic data",
    "quandl", "nasdaq data link",
    "binance", "coinbase", "coingecko", "kraken", "bitfinex", "bitstamp",
    "yfinance",
    "oxford-man institute", "oxford man",
    "realized library",
    "fred-md", "fred-qd",
    "wikipedia", "kaggle",
    "dukascopy", "histdata",
    "macrobond",
    "google trends",
    "investing.com",
    "coinmarketcap", "coinmarketcap.com",
    "cryptocompare", "coindesk", "coinmetrics",
    "akshare", "tushare", "baostock",  # Chinese open-data libraries
    "publicly available",
    "publicly accessible",
    "open access data", "open data",
]

# Disclosed-but-paywalled (counts toward "disclosure" but not "openly accessible")
PAYWALLED_DATA_KEYWORDS = [
    "bloomberg", "reuters", "datastream", "thomson reuters",
    "wrds", "crsp", "compustat",
    "taq", "tick data", "trade and quote", "trades and quotes",
    "optionmetrics", "ivydb",
    "kibot", "kibot.com",
    "cme datamine", "barchart", "tickwrite",
    "olsen", "tickdata",
    "factset",
    "wind database", "wind financial",  # Chinese paywalled
]

MODEL_FAMILIES = OrderedDict([
    ("garch_family", [
        "garch", "egarch", "gjr-garch", "gjr garch", "tgarch", "tarch",
        "fgarch", "igarch", "aparch", "msgarch", "figarch", "realized garch",
    ]),
    ("har_family", ["har-rv", "har rv", "harq", "harx", "har model"]),
    ("ml_family", [
        "random forest", "xgboost", "lightgbm", "gradient boost",
        "neural network", "lstm", "gru", "rnn", "cnn", "transformer",
        "deep learning", "machine learning", "support vector", "svr",
        "ridge regression", "lasso", "elastic net", "boosting",
        "graph neural", "attention",
    ]),
    ("classical", [
        "exponential smoothing", "ewma", "arma", "arima",
        "stochastic volatility", "implied volatility",
    ]),
    ("ensemble", ["ensemble", "stacking", "model averaging", "combination"]),
])

ASSET_KEYWORDS = OrderedDict([
    ("equity", [
        "s&p 500", "s&p500", "sp500", "spx", "nasdaq", "djia", "dow jones",
        "russell", "ftse", "dax", "nikkei", "hang seng", "shanghai composite",
        "stock", "equity", "shares", "stoxx",
    ]),
    ("crypto", [
        "bitcoin", "btc", "ethereum", "eth", "cryptocurrency", "crypto",
        "ripple", "litecoin", "binance", "altcoin",
    ]),
    ("fx", [
        "eur/usd", "gbp/usd", "usd/jpy", "exchange rate", "foreign exchange",
        "fx market", "currency", "forex",
    ]),
    ("commodity", [
        "crude oil", "wti", "brent", "gold", "silver", "natural gas",
        "copper", "futures market", "commodity", "futures",
    ]),
])

WIN_PHRASES = [
    r"(?:we|our results|the results) (?:find|show|demonstrate|indicate|suggest)\b[^.]{5,150}",
    r"(?:best|optimal|superior|outperform[a-z]*|dominant|top-performing)\s+model[^.]{5,150}",
    r"(?:lowest|smallest|minimum)\s+(?:qlike|rmse|mse|mae|loss|error)[^.]{5,150}",
    r"(?:highest|largest|maximum)\s+(?:r[-\s]squared|r2|out-of-sample r)[^.]{5,150}",
]

# --- Helpers -------------------------------------------------------------


def pdftotext(pdf: Path, layout: bool = False) -> str:
    cmd = ["pdftotext"]
    if layout:
        cmd.append("-layout")
    cmd += [str(pdf), "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return r.stdout if r.returncode == 0 else ""
    except Exception as e:
        return f"__PDFTOTEXT_ERROR__ {e}"


def extract_abstract(text: str) -> str:
    """Heuristic: text between 'Abstract' header and first 'Introduction' or 'Keywords:'."""
    lines = text.splitlines()
    start = None
    end = None
    for i, ln in enumerate(lines):
        s = ln.strip().lower()
        if start is None and re.match(r"^\s*abstract\s*$", s):
            start = i + 1
            continue
        if start is None and re.match(r"^\s*abstract[\s:.]", s):
            start = i
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
        # Fallback: first 1000 chars after first paragraph
        return text[:2000].strip()
    if end is None:
        end = min(start + 50, len(lines))
    abstract = " ".join(lines[start:end])
    abstract = re.sub(r"\s+", " ", abstract).strip()
    return abstract[:3000]


CONTEXT_AVAILABILITY_WORDS = [
    "available", "accessible", "deposited", "supplementary", "supplement",
    "replication", "implementation", "code", "scripts", "software",
    "repository", "release", "we provide", "is provided",
    "data and code", "code and data", "open access",
]

CITATION_PATH_PATTERNS = [
    "/wiki/", "/issues/", "/pull/", "/blob/master/README",
    "/blob/main/README", "/discussions/",
]


def find_code_urls(text: str) -> tuple[list[str], list[str]]:
    """Returns (authors_code_urls, all_urls).

    Authors_code_urls require an availability-context word within ±100
    characters of the URL OR appear in a clearly-flagged statement like
    'code is available at <url>'. URLs to citation/wiki pages are demoted.
    """
    all_urls: set[str] = set()
    authors_urls: set[str] = set()
    for pat in CODE_URL_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            url = m.group(0).rstrip(".,;)")
            all_urls.add(url)
            # Skip clearly-citation URLs
            if any(p in url.lower() for p in CITATION_PATH_PATTERNS):
                continue
            # Check ±150-char context
            start = max(0, m.start() - 150)
            end = min(len(text), m.end() + 150)
            context = text[start:end].lower()
            if any(w in context for w in CONTEXT_AVAILABILITY_WORDS):
                authors_urls.add(url)
    return sorted(authors_urls), sorted(all_urls)


def find_public_data_mentions(text: str) -> list[str]:
    text_l = text.lower()
    found = []
    for kw in PUBLIC_DATA_KEYWORDS:
        if kw in text_l:
            found.append(kw)
    return found


def detect_model_families(text: str) -> dict[str, list[str]]:
    text_l = text.lower()
    out: dict[str, list[str]] = {}
    for fam, kws in MODEL_FAMILIES.items():
        hits = sorted({kw for kw in kws if kw in text_l})
        if hits:
            out[fam] = hits
    return out


def detect_assets(text: str) -> list[str]:
    text_l = text.lower()
    found: list[str] = []
    for asset, kws in ASSET_KEYWORDS.items():
        if any(kw in text_l for kw in kws):
            found.append(asset)
    return found


def score_rds(code_urls: list[str], data_mentions: list[str]) -> tuple[int, str, float]:
    """Return (score, reason, confidence)."""
    has_code = bool(code_urls)
    public_data_terms = [d for d in data_mentions if d not in PAYWALLED_DATA_KEYWORDS]
    paywalled_data_terms = [d for d in data_mentions if d in PAYWALLED_DATA_KEYWORDS]
    has_public_data = bool(public_data_terms)
    has_paywalled_disclosed = bool(paywalled_data_terms)

    if has_code and has_public_data:
        return 2, f"code({len(code_urls)})+public_data({len(public_data_terms)})", 0.9
    if has_code and has_paywalled_disclosed:
        return 2, "code+disclosed_paywalled_data (RDS counts disclosure)", 0.7
    if has_code and not data_mentions:
        return 1, "code present but data source unclear (manual review)", 0.4
    if not has_code and has_public_data:
        return 1, f"public_data({len(public_data_terms)}) but no code URL", 0.7
    if not has_code and has_paywalled_disclosed and not has_public_data:
        return 0, "paywalled data only, no code (RDS-0 unless disclosure counts)", 0.6
    return 0, "no code, no public data detected", 0.5


def extract_headline(abstract: str) -> str:
    """Pull the first sentence(s) that name a winner."""
    candidates: list[str] = []
    for pat in WIN_PHRASES:
        for m in re.finditer(pat, abstract, flags=re.IGNORECASE):
            candidates.append(m.group(0).strip())
    return " | ".join(candidates[:3])[:600]


CANONICAL_HEADINGS = [
    "abstract", "introduction", "literature review", "related work",
    "data", "data and methodology", "methodology", "methods",
    "model", "models", "the model", "empirical model", "econometric model",
    "results", "empirical results", "main results",
    "robustness", "robustness checks", "robustness check",
    "discussion", "conclusion", "conclusions", "concluding remarks",
    "appendix", "references", "acknowledg", "supplementary",
    "data availability", "code availability",
]


def extract_section_headings(text: str) -> list[str]:
    """Detect section headings via:
    (1) numbered "1. Introduction" pattern
    (2) title-case standalone lines that match canonical heading words
    """
    headings = []
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s or len(s) > 100:
            continue
        # Numbered headings: "1. Introduction" or "1 Introduction"
        m = re.match(r"^([1-9]\d?)(?:\.\d+)*\.?\s+([A-Z][A-Za-z\-\s,&]{3,80})\s*$", s)
        if m:
            num = m.group(1)
            title = m.group(2).strip().rstrip(".")
            if not title.lower().startswith(("the ", "a ", "an ")) or len(title) < 30:
                headings.append(f"{num}. {title}")
                continue
        # Plain title-case canonical heading
        s_lower = s.lower().rstrip(".:")
        if any(s_lower == h or s_lower.startswith(h + " ") for h in CANONICAL_HEADINGS):
            # Check it's standalone-ish (preceding/following line blank or short)
            prev_blank = i == 0 or not lines[i - 1].strip()
            short_line = len(s) <= 60
            if prev_blank or short_line:
                headings.append(s.rstrip(":."))
    seen = set()
    out = []
    for h in headings:
        key = h.lower().strip()
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out[:40]


def detect_repro_statement(text: str) -> str:
    """Look for a 'data and code available' style sentence."""
    text_l = text.lower()
    patterns = [
        r"(?:code|software|programs?|scripts?)\s+(?:and|&)\s+(?:data|datasets?|replication[\w\s]*)\s+(?:are|is|will be|can be)\s+(?:available|accessible|provided|deposited)[^.]{5,300}",
        r"(?:data|code|software)\s+(?:are|is)\s+(?:available|accessible)\s+(?:at|on|via|from|through)\s+[^.]{5,300}",
        r"replication\s+(?:files?|materials?|package|kit)\s+[^.]{5,300}",
        r"(?:available|accessible)\s+(?:on|at|via)\s+(?:github|gitlab|zenodo|osf|figshare|the journal[\'\\s]*s? website)[^.]{5,200}",
    ]
    for pat in patterns:
        m = re.search(pat, text_l, flags=re.IGNORECASE)
        if m:
            # Return the matched text from the original (case-preserved)
            start, end = m.span()
            return re.sub(r"\s+", " ", text[start:end]).strip()[:400]
    return ""


def count_tables_figures(text: str) -> tuple[int, int]:
    """Count distinct 'Table N' and 'Figure N' references in main text."""
    tables = set(re.findall(r"\bTable\s+(\d{1,2})\b", text))
    figures = set(re.findall(r"\bFigure\s+(\d{1,2})\b", text))
    return len(tables), len(figures)


# --- Main ----------------------------------------------------------------


def process_paper(row: dict) -> dict:
    pid = row["paper_id"]
    pdf = PER_PAPER / pid / "source.pdf"
    out: dict = {
        "paper_id": pid,
        "title": row["title"],
        "year": row["year"],
        "venue": row["venue"],
        "citations": int(row.get("citations") or 0),
        "doi": row["doi"],
        "arxiv_id": row.get("arxiv_id", ""),
    }
    if not pdf.exists():
        out["error"] = "no PDF"
        return out

    text = pdftotext(pdf, layout=False)
    if text.startswith("__PDFTOTEXT_ERROR__"):
        out["error"] = text
        return out
    text_layout = pdftotext(pdf, layout=True)

    # Abstract
    abstract = extract_abstract(text)
    out["abstract"] = abstract
    out["abstract_word_count"] = len(abstract.split())

    # Code & data
    authors_code_urls, all_code_urls = find_code_urls(text)
    data_mentions = find_public_data_mentions(text)
    out["code_urls"] = authors_code_urls  # used for RDS scoring
    out["all_code_urls"] = all_code_urls  # all GitHub-style URLs (incl. citations)
    out["data_source_mentions"] = data_mentions

    # RDS
    rds, reason, conf = score_rds(authors_code_urls, data_mentions)
    out["rds_score"] = rds
    out["rds_reason"] = reason
    out["rds_confidence"] = conf
    out["needs_review"] = conf < 0.7

    # Models compared
    families = detect_model_families(text)
    out["model_families"] = families
    out["n_model_families"] = len(families)

    # Assets
    out["asset_classes"] = detect_assets(text)

    # Headline winner phrases
    out["headline_phrases"] = extract_headline(abstract)

    # Reproducibility statement
    out["reproducibility_statement"] = detect_repro_statement(text)

    # Tables & figures
    n_tab, n_fig = count_tables_figures(text)
    out["n_tables"] = n_tab
    out["n_figures"] = n_fig

    # Section structure
    out["section_headings"] = extract_section_headings(text)
    out["n_sections_detected"] = len(out["section_headings"])

    # PDF size
    out["pdf_size_kb"] = pdf.stat().st_size // 1024

    # Persist per-paper JSON
    json_path = pdf.parent / "metadata.json"
    json_path.write_text(json.dumps(out, indent=2, default=str))

    return out


def main() -> int:
    p = ROOT / "candidate_30.csv"
    with p.open() as f:
        rows = list(csv.DictReader(f))
    print(f"Processing {len(rows)} papers...\n")

    results: list[dict] = []
    for r in rows:
        out = process_paper(r)
        results.append(out)
        title_short = (out.get("title") or "")[:55]
        rds = out.get("rds_score", "?")
        nfam = out.get("n_model_families", 0)
        nrev = "REVIEW" if out.get("needs_review") else "ok"
        print(f"  {out['paper_id']:30}  RDS={rds}  fam={nfam}  [{nrev}]  {title_short}")
        if out.get("error"):
            print(f"    ERROR: {out['error']}")

    # Aggregate CSV
    csv_path = RESULTS / "phase2_per_paper_metadata.csv"
    fieldnames = [
        "paper_id", "year", "venue", "citations", "doi", "arxiv_id",
        "abstract_word_count", "n_model_families", "model_families_summary",
        "asset_classes", "rds_score", "rds_reason", "rds_confidence",
        "needs_review", "n_code_urls", "code_urls_first",
        "n_data_mentions", "data_mentions_summary", "headline_phrases",
        "n_tables", "n_figures", "n_sections_detected", "pdf_size_kb",
        "title",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {
                "paper_id": r["paper_id"],
                "year": r.get("year", ""),
                "venue": r.get("venue", ""),
                "citations": r.get("citations", 0),
                "doi": r.get("doi", ""),
                "arxiv_id": r.get("arxiv_id", ""),
                "abstract_word_count": r.get("abstract_word_count", 0),
                "n_model_families": r.get("n_model_families", 0),
                "model_families_summary": ",".join(
                    f"{k}({len(v)})" for k, v in (r.get("model_families") or {}).items()
                ),
                "asset_classes": ",".join(r.get("asset_classes", [])),
                "rds_score": r.get("rds_score", ""),
                "rds_reason": r.get("rds_reason", ""),
                "rds_confidence": r.get("rds_confidence", ""),
                "needs_review": r.get("needs_review", False),
                "n_code_urls": len(r.get("code_urls", [])),
                "code_urls_first": (r.get("code_urls", [""]) or [""])[0],
                "n_data_mentions": len(r.get("data_source_mentions", [])),
                "data_mentions_summary": ",".join(r.get("data_source_mentions", [])[:5]),
                "headline_phrases": (r.get("headline_phrases", "") or "")[:300],
                "n_tables": r.get("n_tables", 0),
                "n_figures": r.get("n_figures", 0),
                "n_sections_detected": r.get("n_sections_detected", 0),
                "pdf_size_kb": r.get("pdf_size_kb", 0),
                "title": r.get("title", ""),
            }
            w.writerow(row)

    # Summary
    n = len(results)
    n_rds2 = sum(1 for r in results if r.get("rds_score") == 2)
    n_rds1 = sum(1 for r in results if r.get("rds_score") == 1)
    n_rds0 = sum(1 for r in results if r.get("rds_score") == 0)
    n_review = sum(1 for r in results if r.get("needs_review"))
    n_with_code = sum(1 for r in results if r.get("code_urls"))

    print("\n=== Summary ===")
    print(f"  Total: {n}")
    print(f"  RDS-2: {n_rds2}  ({100*n_rds2/n:.0f}%)")
    print(f"  RDS-1: {n_rds1}  ({100*n_rds1/n:.0f}%)")
    print(f"  RDS-0: {n_rds0}  ({100*n_rds0/n:.0f}%)")
    print(f"  With code URL detected: {n_with_code}")
    print(f"  Flagged for human review: {n_review}")
    print(f"\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
