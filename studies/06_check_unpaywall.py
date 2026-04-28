"""Phase 2 step 3: check Unpaywall for free legal copies of paywalled papers.

Unpaywall (a project by OurResearch, used by university libraries worldwide)
indexes legitimate open-access versions of paywalled papers — author preprints
on institutional repositories, SSRN deposits, accepted manuscripts, etc.

Reads:  studies/candidate_30.csv  (the 30 chosen papers, with DOI column)
Writes: studies/candidate_30_access.csv  (adds best_oa_url + oa_status columns)

API: https://api.unpaywall.org/v2/{doi}?email=...
Free, no key, polite rate ≤100k req/day.
"""
from __future__ import annotations

import csv
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

try:
    import certifi

    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "candidate_30.csv"
OUTPUT = ROOT / "candidate_30_access.csv"
EMAIL = "1819ak@gmail.com"
USER_AGENT = f"vol-eval-survey/0.1 (mailto:{EMAIL})"


def http_get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as r:
        return r.read()


def query_unpaywall(doi: str) -> dict:
    if not doi:
        return {"oa_status": "n/a (no DOI)", "best_oa_url": "", "best_oa_host": ""}
    url = (
        "https://api.unpaywall.org/v2/"
        + urllib.parse.quote(doi, safe="")
        + f"?email={urllib.parse.quote(EMAIL)}"
    )
    try:
        payload = json.loads(http_get(url))
    except urllib.error.HTTPError as e:
        return {"oa_status": f"http_{e.code}", "best_oa_url": "", "best_oa_host": ""}
    except Exception as e:
        return {"oa_status": f"error:{e}", "best_oa_url": "", "best_oa_host": ""}

    oa_status = payload.get("oa_status", "unknown")
    best = payload.get("best_oa_location") or {}
    locations = payload.get("oa_locations") or []
    return {
        "oa_status": oa_status,
        "is_oa": payload.get("is_oa", False),
        "best_oa_url": best.get("url_for_pdf") or best.get("url") or "",
        "best_oa_host": (best.get("host_type") or ""),
        "n_oa_locations": len(locations),
        "all_oa_urls": "; ".join(
            (loc.get("url_for_pdf") or loc.get("url") or "") for loc in locations
        )[:500],
    }


def main() -> int:
    if not INPUT.exists():
        print(
            f"ERROR: {INPUT} not found. Create it first (the 30 chosen papers).",
            file=sys.stderr,
        )
        return 1

    with INPUT.open() as f:
        rows = list(csv.DictReader(f))

    print(f"Checking {len(rows)} papers via Unpaywall API...\n")

    out_rows: list[dict] = []
    for i, r in enumerate(rows, 1):
        doi = (r.get("doi") or "").strip()
        result = query_unpaywall(doi)
        merged = {**r, **result}
        out_rows.append(merged)
        status = result["oa_status"]
        title = (r.get("title") or "")[:70]
        marker = "FREE" if result.get("is_oa") else "    "
        print(f"  {i:>2}. [{marker} {status:>10}] {title}")
        if result.get("best_oa_url"):
            print(f"        -> {result['best_oa_url']}")
        time.sleep(0.15)

    fieldnames = list(rows[0].keys()) + [
        "oa_status",
        "is_oa",
        "best_oa_url",
        "best_oa_host",
        "n_oa_locations",
        "all_oa_urls",
    ]
    seen = set()
    fn_unique = [x for x in fieldnames if not (x in seen or seen.add(x))]

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn_unique, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    n_free = sum(1 for r in out_rows if r.get("is_oa"))
    print(
        f"\nResult: {n_free} of {len(out_rows)} have a free legal copy via Unpaywall."
    )
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
