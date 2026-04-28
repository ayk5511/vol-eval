"""Phase 2 step 4: download the free PDFs.

Reads:  studies/candidate_30_access.csv
Writes: studies/per_paper/<paper_id>/source.pdf  for every paper with a free
        legal URL (arXiv preprint, institutional repository, hybrid OA, etc.).

For papers with neither a DOI-linked free copy nor an arxiv_id, prints the
publisher URL the user must hit through their library access.

Run: python3 studies/07_download_papers.py
"""
from __future__ import annotations

import csv
import ssl
import subprocess
import sys
import time
from pathlib import Path

try:
    import certifi

    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "candidate_30_access.csv"
PER_PAPER = ROOT / "per_paper"
PER_PAPER.mkdir(exist_ok=True)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def download_curl(url: str, dest: Path, timeout: int = 90) -> tuple[bool, str]:
    """Use curl with full browser-like headers (handles gzip/brotli + Sec-Fetch).
    MDPI and similar sites require this; urllib alone gets 403.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl",
        "-sL",
        "--max-time",
        str(timeout),
        "--compressed",
        "-H",
        f"User-Agent: {USER_AGENT}",
        "-H",
        "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.8,*/*;q=0.5",
        "-H",
        "Accept-Language: en-US,en;q=0.9",
        "-H",
        "Sec-Fetch-Dest: document",
        "-H",
        "Sec-Fetch-Mode: navigate",
        "-H",
        "Sec-Fetch-Site: none",
        "-o",
        str(dest),
        "-w",
        "%{http_code}|%{size_download}",
        url,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
        if r.returncode != 0:
            return False, f"curl rc={r.returncode}: {r.stderr.strip()[:80]}"
        out = r.stdout.strip()
        code, size = out.split("|", 1) if "|" in out else (out, "0")
        if code != "200":
            return False, f"http_{code}"
        if not dest.exists() or dest.stat().st_size < 1024:
            return False, f"empty (got {size} bytes)"
        with dest.open("rb") as f:
            if f.read(4) != b"%PDF":
                return False, f"non-PDF (got {size} bytes)"
        return True, f"OK ({int(size) // 1024}KB)"
    except subprocess.TimeoutExpired:
        return False, "curl timeout"
    except Exception as e:
        return False, f"error: {e}"


def download(url: str, dest: Path, timeout: int = 90) -> tuple[bool, str]:
    if dest.exists() and dest.stat().st_size > 1024:
        with dest.open("rb") as f:
            head = f.read(4)
        if head == b"%PDF":
            return True, f"skip (exists, {dest.stat().st_size // 1024}KB)"
    return download_curl(url, dest, timeout)


def main() -> int:
    if not INPUT.exists():
        print(f"ERROR: {INPUT} not found. Run 06_check_unpaywall.py first.", file=sys.stderr)
        return 1

    with INPUT.open() as f:
        rows = list(csv.DictReader(f))

    print(f"Processing {len(rows)} papers...\n")

    n_ok = 0
    needs_user: list[dict] = []
    for r in rows:
        pid = r["paper_id"]
        dest = PER_PAPER / pid / "source.pdf"

        # Pre-check for existing valid PDF (e.g., manually placed or fetched
        # via a different path like Semantic Scholar)
        if dest.exists() and dest.stat().st_size > 1024:
            with dest.open("rb") as f:
                if f.read(4) == b"%PDF":
                    print(f"  {pid}: OK   [pre-existing]  ({dest.stat().st_size // 1024}KB)")
                    n_ok += 1
                    continue

        url = ""
        source = ""

        arxiv_id = (r.get("arxiv_id") or "").strip()
        oa_url = (r.get("best_oa_url") or "").strip()
        doi = (r.get("doi") or "").strip()

        if arxiv_id:
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            source = "arxiv"
        elif oa_url and r.get("is_oa") in ("True", "true", True, "1"):
            url = oa_url
            source = f"unpaywall:{r.get('best_oa_host', '')}"

        if not url:
            needs_user.append(r)
            print(f"  {pid}: NEEDS USER ACCESS  ({r.get('venue', '')})")
            continue

        ok, msg = download(url, dest)
        marker = "OK  " if ok else "FAIL"
        print(f"  {pid}: {marker} [{source}]  {msg}")
        if ok:
            n_ok += 1
        else:
            needs_user.append(r)
        time.sleep(1.0)  # polite

    print(f"\nDownloaded: {n_ok} of {len(rows)}")
    print(f"Needs user access: {len(needs_user)}")
    if needs_user:
        print("\nPapers requiring your library access:")
        for r in needs_user:
            print(f"  {r['paper_id']:30}  {r['venue']:35}  {r['url']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
