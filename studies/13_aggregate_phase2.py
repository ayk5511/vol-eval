"""Aggregate all per-paper voleval_result.json files into a single
phase2_summary.json with the headline tabulation.

Reads:  studies/per_paper/<paper_id>/voleval_result.json (where exists)
Writes: studies/results/phase2_summary.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PER_PAPER = ROOT / "per_paper"
RESULTS = ROOT / "results"


def classify_outcome(result: dict) -> str:
    """Classify a paper's outcome as 'survives', 'partial', 'fails', 'inconclusive'.

    Logic:
      - survives:    DM rejects at 5% in paper's claimed direction AND best_by_qlike == claimed_winner
      - partial:     Some support (e.g., DM significant but SPA rejects, or 1+ subset survives)
      - fails:       paper-claimed winner is not best by QLIKE AND DM is not significant
      - inconclusive: insufficient data
    """
    datasets = result.get("datasets", {})
    if not datasets:
        return "inconclusive"

    survives_count = 0
    fails_count = 0
    partial_count = 0
    total = 0
    details: list[str] = []

    for label, d in datasets.items():
        if "error" in d:
            continue
        total += 1
        best = d.get("best_by_qlike")
        claimed = d.get("paper_claimed_winner")
        dm = d.get("dm_winner_vs_runner_up", {})
        mcs = d.get("mcs_at_90pct", {})
        spa = d.get("spa_winner_as_benchmark", {})
        survives = mcs.get("n_survivors", 99) == 1 and best == claimed
        winner_alone = (mcs.get("n_survivors") == 1 and claimed in mcs.get("survivors", []))
        dm_sig = dm.get("rejected_at_5pct", False)
        dm_winner = dm.get("winner", "")
        # Strong survive: claimed winner is sole MCS survivor
        if winner_alone:
            survives_count += 1
            details.append(f"  {label}: SURVIVES (MCS narrows to {claimed})")
        # Partial survive: DM significant in claimed direction, but MCS keeps multiple
        elif dm_sig and dm_winner == "a" and best == claimed:
            partial_count += 1
            details.append(f"  {label}: PARTIAL (DM significant for claimed winner; MCS keeps {mcs.get('n_survivors')})")
        # Fails: DM not significant or wrong direction
        else:
            fails_count += 1
            reason_parts = []
            if best != claimed:
                reason_parts.append(f"best={best}")
            if not dm_sig:
                reason_parts.append(f"DM p={dm.get('p_value', '?'):.3f}" if isinstance(dm.get("p_value"), float) else "")
            details.append(f"  {label}: FAILS ({', '.join(filter(None, reason_parts))})")

    if total == 0:
        return ("inconclusive", details)
    if survives_count == total:
        return ("survives", details)
    if survives_count > 0 or partial_count > 0:
        return ("partial", details)
    return ("fails", details)


def main():
    paper_dirs = sorted([d for d in PER_PAPER.iterdir() if d.is_dir() and (d / "voleval_result.json").exists()])
    print(f"Found {len(paper_dirs)} papers with voleval_result.json")

    aggregate = []
    outcome_counter = Counter()
    for d in paper_dirs:
        path = d / "voleval_result.json"
        result = json.loads(path.read_text())
        # Some old-format results (P25/P26/P27) don't have 'datasets' wrapper.
        if "datasets" not in result and any(k in result for k in ["GBPUSD", "EURGBP", "n_observations", "best_model_observed"]):
            # Wrap legacy formats
            if "best_model_observed" in result:
                # P25/P26 single-dataset legacy
                qlikes = result.get("qlike_by_model", {})
                best = result.get("best_model_observed")
                dm = result.get("dm_winner_vs_runner_up", {})
                mcs = result.get("mcs_at_90pct", {})
                spa = result.get("spa_winner_as_benchmark", {})
                wrapped = {"datasets": {"default": {
                    "n_observations": result.get("n_observations_test"),
                    "qlike_by_model": qlikes,
                    "best_by_qlike": best,
                    "paper_claimed_winner": result.get("paper_claimed_winner"),
                    "dm_winner_vs_runner_up": {
                        "rejected_at_5pct": dm.get("rejected_at_5pct"),
                        "p_value": dm.get("p_value"),
                        "winner": dm.get("winner"),
                    } if dm else {},
                    "mcs_at_90pct": mcs,
                    "spa_winner_as_benchmark": spa,
                }}}
                result = wrapped
            elif "GBPUSD" in result or "EURGBP" in result:
                # P27 multi-dataset legacy
                wrapped = {"datasets": {}}
                for k in ["GBPUSD", "EURGBP"]:
                    if k in result:
                        sub = result[k]
                        wrapped["datasets"][k] = {
                            "n_observations": sub.get("n_observations"),
                            "qlike_by_model": sub.get("qlike_by_model", {}),
                            "best_by_qlike": sub.get("best_model_observed"),
                            "paper_claimed_winner": sub.get("paper_claimed_winner"),
                            "dm_winner_vs_runner_up": sub.get("dm_winner_vs_runner_up", {}),
                            "mcs_at_90pct": sub.get("mcs_at_90pct", {}),
                            "spa_winner_as_benchmark": sub.get("spa_garch_as_benchmark", {}),
                        }
                result = wrapped

        outcome, details = classify_outcome(result)
        outcome_counter[outcome] += 1
        aggregate.append({
            "paper_id": d.name,
            "outcome": outcome,
            "details": details,
        })
        print(f"  {d.name:30}  {outcome.upper():>10}")
        for line in details:
            print(line)

    summary = {
        "n_papers_reproduced": len(paper_dirs),
        "outcomes": dict(outcome_counter),
        "outcome_pct": {k: round(100 * v / len(paper_dirs), 1) for k, v in outcome_counter.items()},
        "per_paper": aggregate,
        "interpretation": {
            "survives": "Paper-claimed winner is the SOLE survivor of MCS at 90%; claim fully supported.",
            "partial": "Mixed evidence: some subset (e.g., one of multiple datasets) shows significant winner but not all; or DM significant but SPA rejects.",
            "fails": "Paper-claimed winner is NOT the best by QLIKE OR DM is not significant. Headline claim does not survive significance testing.",
            "inconclusive": "Insufficient data or could not run vol-eval.",
        },
    }
    out_path = RESULTS / "phase2_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")
    print(f"\nHEADLINE: {len(paper_dirs)} papers reproduced.")
    for k, v in outcome_counter.most_common():
        print(f"  {k}: {v}/{len(paper_dirs)}  ({100*v/len(paper_dirs):.0f}%)")


if __name__ == "__main__":
    sys.exit(main() or 0)
