"""Phase 2 step 2: narrow ~400 loosely-filtered candidates down to ~50-70 strong
horse-race papers, stratified across venues, ready for the user's final 30.

Reads:  studies/results/candidates_filtered.csv
Writes: studies/results/candidates_top.csv

Stricter horse-race criterion:
- Title or abstract mentions ≥2 distinct model families (GARCH-family / HAR /
  ML / classical), OR mentions ≥1 family + an explicit comparison statistic
  (DM / MCS / SPA / Reality Check)
- Volatility must be the predictand (excludes portfolio / option-pricing /
  return-prediction-only papers)
- Hard exclusions on review and off-topic title patterns

Stratified sampling: top-N per venue by citations, capped to keep the candidate
list manageable for a human skim.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

INPUT = RESULTS / "candidates_filtered.csv"
OUTPUT = RESULTS / "candidates_top.csv"

MODEL_FAMILIES = {
    "garch": (
        "garch", " egarch", "gjr-garch", "gjr garch", "tgarch", "tarch",
        "fgarch", "igarch", "aparch", "msgarch", "figarch",
    ),
    "har": ("har-rv", "har rv", "harq", "harx", " har "),
    "ml": (
        "random forest", "xgboost", "lightgbm", "gradient boost",
        "neural network", "lstm", "gru", " rnn ", " cnn ", "transformer",
        "deep learning", "machine learning", "support vector", " svm",
        "ridge regression", "lasso ", "elastic net", "tree-based", "boosting",
        "attention", "graph neural", "spatial-temporal", "spatial–temporal",
        "encoder-decoder", "autoencoder", "reinforcement learning",
    ),
    "classical": (
        "exponential smoothing", "ewma", "arma", "arima", "ar(1)", "ar-1",
        "stochastic volatility", "realized garch", "implied volatility",
        "rolling window", "ols ", " ols",
    ),
    "ensemble": ("ensemble", "stacking", "bagging", "model averaging", "combination"),
}

EXPLICIT_TEST_TERMS = (
    "diebold-mariano", "diebold mariano", "model confidence set",
    "superior predictive ability", "reality check", "white test",
    "hansen test", " mcs ", " spa ", " dm test", "encompassing",
    "model selection", "false discovery rate", "model comparison",
    "out-of-sample r", "qlike", "out-of-sample loss",
)

VOL_PREDICTAND_TERMS = (
    "volatility forecast", "volatility prediction", "realized volatility",
    "realized variance", "rv forecast", "vix forecast", "implied volatility",
    "conditional variance", "variance forecast", "volatility model",
)

HARD_EXCLUDE_TITLE = (
    "portfolio optimization", "portfolio choice", "asset allocation",
    "option pricing", "perspective", "literature", "survey", "review",
    "tutorial", "introduction to", "overview of", "guide to",
)

HARD_EXCLUDE_ABSTRACT = (
    "portfolio optimization", "asset allocation problem",
    "we propose a new portfolio", "investor's portfolio",
)

# Per-venue caps for the final shortlist
VENUE_CAPS = {
    "Journal of Financial Econometrics": 12,
    "Journal of Empirical Finance": 8,
    "Quantitative Finance": 8,
    "Journal of Forecasting": 12,
    "International Journal of Forecasting": 8,
    "Journal of Financial Data Science": 4,
    "arXiv q-fin": 18,
}


def detect_families(text: str) -> set[str]:
    found: set[str] = set()
    for family, terms in MODEL_FAMILIES.items():
        if any(t in text for t in terms):
            found.add(family)
    return found


def has_explicit_test(text: str) -> bool:
    return any(t in text for t in EXPLICIT_TEST_TERMS)


def has_vol_predictand(text: str) -> bool:
    return any(t in text for t in VOL_PREDICTAND_TERMS)


def is_hard_excluded(title: str, abstract: str) -> str | None:
    title_l = title.lower()
    if any(t in title_l for t in HARD_EXCLUDE_TITLE):
        return "title_excluded"
    abs_l = abstract.lower()
    if any(t in abs_l for t in HARD_EXCLUDE_ABSTRACT):
        return "abstract_excluded"
    return None


def score(row: dict) -> tuple[int, int, str]:
    """Strength score for a candidate. Higher = stronger horse-race signal.
    Returns (score, citations, reason). score==0 means reject.
    """
    title = row.get("title", "") or ""
    abstract = row.get("abstract", "") or ""
    text = (title + " " + abstract).lower()

    excluded = is_hard_excluded(title, abstract)
    if excluded:
        return (0, 0, excluded)

    if not has_vol_predictand(text):
        return (0, 0, "no_vol_predictand")

    families = detect_families(text)
    has_test = has_explicit_test(text)
    title_l = title.lower()
    horse_race_in_title = any(t in title_l for t in (
        "horse race", "horse-race", "comparison", "compare ",
        "model selection", "competing", "versus", " vs ", " vs.",
    ))

    # Tiered scoring (higher = stronger horse-race signal)
    if len(families) >= 2 and has_test:
        s, reason = 4, f"families={len(families)}+test"
    elif len(families) >= 2:
        s, reason = 3, f"families={len(families)}:{','.join(sorted(families))}"
    elif len(families) == 1 and has_test:
        s, reason = 2, "family+test"
    elif len(families) == 1 and horse_race_in_title:
        s, reason = 2, "family+horse_race_title"
    elif has_test:
        s, reason = 1, "test_only"
    elif len(families) == 1:
        s, reason = 1, f"family_only:{','.join(families)}"
    else:
        return (0, 0, "no_family_no_test")

    cites = int(row.get("citations") or 0)
    return (s, cites, reason)


def main() -> int:
    if not INPUT.exists():
        print(f"ERROR: {INPUT} not found. Run 04_search_candidates.py first.", file=sys.stderr)
        return 1

    with INPUT.open() as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} loosely-filtered candidates")

    scored: list[dict] = []
    rejected_by_reason: dict[str, int] = defaultdict(int)
    for r in rows:
        s, cites, reason = score(r)
        if s == 0:
            rejected_by_reason[reason] += 1
            continue
        r2 = dict(r)
        r2["score"] = s
        r2["citations_int"] = cites
        r2["score_reason"] = reason
        scored.append(r2)

    print(f"\nPassed strict criterion: {len(scored)}")
    print("Rejected by reason:")
    for reason, n in sorted(rejected_by_reason.items(), key=lambda x: -x[1]):
        print(f"  {n:>4}  {reason}")

    # Stratified by venue: keep top by (score, citations) per venue
    by_venue: dict[str, list[dict]] = defaultdict(list)
    for r in scored:
        by_venue[r.get("venue", "")].append(r)

    final: list[dict] = []
    for venue, items in by_venue.items():
        items.sort(key=lambda x: (-x["score"], -x["citations_int"]))
        cap = VENUE_CAPS.get(venue, 4)
        kept = items[:cap]
        final.extend(kept)
        print(f"  {venue}: {len(items)} qualified, kept top {len(kept)}")

    final.sort(key=lambda x: (-x["score"], -x["citations_int"], x.get("year") or 0))

    fieldnames = [
        "score", "citations_int", "score_reason",
        "source", "doi", "title", "abstract", "venue", "year",
        "citations", "authors", "url",
    ]
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in final:
            w.writerow(r)

    print(f"\nWrote {OUTPUT}  ({len(final)} rows)")
    print("\nTop 15 by (score, citations):")
    for r in final[:15]:
        title = r["title"][:75]
        venue_short = r["venue"][:25]
        print(f"  s={r['score']} c={r['citations_int']:>4} {r['year']} | {venue_short:25} | {title}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
