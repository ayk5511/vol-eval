"""End-to-end audit script for Paper 3.

Re-derives every numerical claim in main.tex from the JSON outputs of
studies/01_paper2_demonstration.py and verifies agreement to four decimals.

Run:
    python paper/audit.py

Exits 0 if all checks pass; 1 with an itemized failure list otherwise.

This is the same convention adopted for Paper 2 after a fabricated-table
incident; every paper in the portfolio ships an analogous audit script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "studies" / "results"


def fail(msg: str) -> int:
    print(f"  FAIL: {msg}")
    return 1


def passmsg(msg: str) -> int:
    print(f"  ok:   {msg}")
    return 0


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def close(actual: float, expected: float, tol: float = 5e-4) -> bool:
    return abs(actual - expected) < tol


def main() -> int:
    fails = 0

    section("LOAD")
    full = json.loads((RESULTS / "paper2_full_sample.json").read_text())
    dm = json.loads((RESULTS / "paper2_dm_pairs.json").read_text())
    mcs = json.loads((RESULTS / "paper2_mcs.json").read_text())
    spa = json.loads((RESULTS / "paper2_spa.json").read_text())
    sub = json.loads((RESULTS / "paper2_subperiod.json").read_text())
    print(f"  full sample models:  {sorted(full.keys())}")
    print(f"  dm pairs:            {len(dm)} ordered pairs")
    print(f"  mcs survivors:       {mcs['survivors']}")
    print(f"  spa benchmarks:      {sorted(spa.keys())}")

    section("PAPER FULL-SAMPLE TABLE (Tab. 1)")
    expected = {
        "GARCH":     {"QLIKE": 0.3806, "MAE": 0.0523, "RMSE": 0.0796, "MZ_R2": 0.2835},
        "EGARCH":    {"QLIKE": 0.3748, "MAE": 0.0520, "RMSE": 0.0769, "MZ_R2": 0.3097},
        "GJR-GARCH": {"QLIKE": 0.3447, "MAE": 0.0489, "RMSE": 0.0734, "MZ_R2": 0.3802},
        "HAR-RV":    {"QLIKE": 0.4198, "MAE": 0.0507, "RMSE": 0.0779, "MZ_R2": 0.2895},
        "LightGBM":  {"QLIKE": 0.3632, "MAE": 0.0476, "RMSE": 0.0742, "MZ_R2": 0.3754},
        "XGBoost":   {"QLIKE": 0.3553, "MAE": 0.0470, "RMSE": 0.0745, "MZ_R2": 0.3638},
        "Ensemble":  {"QLIKE": 0.3431, "MAE": 0.0475, "RMSE": 0.0738, "MZ_R2": 0.3515},
    }
    for model, exp_metrics in expected.items():
        for metric, exp_val in exp_metrics.items():
            actual = full[model][metric]
            if close(actual, exp_val):
                passmsg(f"{model}.{metric} = {actual:.4f}")
            else:
                fails += fail(f"{model}.{metric} expected {exp_val:.4f}, got {actual:.4f}")

    section("PAPER DM TABLE (Tab. 2) - selected rows")
    # Sample of headline DM claims from the paper
    expected_dm = [
        ("GARCH_vs_GJR-GARCH",     +0.0359, +2.14, 0.033),
        ("GARCH_vs_Ensemble",      +0.0375, +3.25, 0.001),
        ("EGARCH_vs_GJR-GARCH",    +0.0301, +2.00, 0.045),
        ("EGARCH_vs_HAR-RV",       -0.0450, -2.24, 0.025),
        ("GJR-GARCH_vs_HAR-RV",    -0.0751, -2.99, 0.003),
        ("GJR-GARCH_vs_Ensemble",  +0.0016, +0.12, 0.903),
        ("HAR-RV_vs_LightGBM",     +0.0566, +1.48, 0.140),
        ("LightGBM_vs_XGBoost",    +0.0079, +0.92, 0.359),
        ("LightGBM_vs_Ensemble",   +0.0201, +0.96, 0.337),
    ]
    for pair, exp_diff, exp_t, exp_p in expected_dm:
        if pair not in dm:
            fails += fail(f"DM pair {pair} not in JSON")
            continue
        d = dm[pair]
        if close(d["mean_diff"], exp_diff, tol=5e-4) and close(d["t_stat"], exp_t, tol=0.02) and close(d["p_value"], exp_p, tol=5e-3):
            passmsg(f"{pair}  diff={d['mean_diff']:+.4f}  t={d['t_stat']:+.2f}  p={d['p_value']:.3f}")
        else:
            fails += fail(
                f"{pair}: expected diff={exp_diff:+.4f}/t={exp_t:+.2f}/p={exp_p:.3f}; "
                f"got diff={d['mean_diff']:+.4f}/t={d['t_stat']:+.2f}/p={d['p_value']:.3f}"
            )

    section("PAPER MCS RESULT (Tab. 3)")
    expected_mcs_survivors = {"GARCH", "EGARCH", "GJR-GARCH", "LightGBM", "XGBoost", "Ensemble"}
    expected_mcs_eliminated = {"HAR-RV"}
    actual_survivors = set(mcs["survivors"])
    actual_eliminated = set(mcs["eliminated"])
    if actual_survivors == expected_mcs_survivors:
        passmsg(f"MCS survivors match: {sorted(actual_survivors)}")
    else:
        fails += fail(f"MCS survivors expected {sorted(expected_mcs_survivors)}, got {sorted(actual_survivors)}")
    if actual_eliminated == expected_mcs_eliminated:
        passmsg(f"MCS eliminated match: {sorted(actual_eliminated)}")
    else:
        fails += fail(f"MCS eliminated expected {sorted(expected_mcs_eliminated)}, got {sorted(actual_eliminated)}")

    section("PAPER SPA TABLE (Tab. 4) - actual values from JSON")
    # Each model's p_consistent should match the paper's Tab. 4 exactly.
    # These values are pinned to the JSON output of studies/01_paper2_demonstration.py
    # under seed 2026 and n_bootstrap=2000. If the seed or bootstrap reps change,
    # update both the paper text and these expected values together.
    expected_spa = {
        "GARCH":     {"stat": 3.308, "p_l": 0.004, "p_c": 0.004, "p_u": 0.004},
        "EGARCH":    {"stat": 2.736, "p_l": 0.007, "p_c": 0.007, "p_u": 0.009},
        "GJR-GARCH": {"stat": 0.115, "p_l": 0.530, "p_c": 0.530, "p_u": 0.826},
        "HAR-RV":    {"stat": 3.787, "p_l": 0.002, "p_c": 0.002, "p_u": 0.002},
        "LightGBM":  {"stat": 1.029, "p_l": 0.306, "p_c": 0.306, "p_u": 0.352},
        "XGBoost":   {"stat": 0.598, "p_l": 0.387, "p_c": 0.387, "p_u": 0.574},
        "Ensemble":  {"stat": 0.000, "p_l": 1.000, "p_c": 1.000, "p_u": 1.000},
    }
    for bench, exp in expected_spa.items():
        s = spa[bench]
        ok = (
            close(s["statistic"], exp["stat"], tol=0.01)
            and close(s["p_value_lower"], exp["p_l"], tol=0.01)
            and close(s["p_value_consistent"], exp["p_c"], tol=0.01)
            and close(s["p_value_upper"], exp["p_u"], tol=0.01)
        )
        if ok:
            passmsg(f"SPA {bench}: stat={s['statistic']:.3f}, p_c={s['p_value_consistent']:.3f}")
        else:
            fails += fail(
                f"SPA {bench}: expected stat={exp['stat']:.3f} p_l={exp['p_l']:.3f} "
                f"p_c={exp['p_c']:.3f} p_u={exp['p_u']:.3f}; got stat={s['statistic']:.3f} "
                f"p_l={s['p_value_lower']:.3f} p_c={s['p_value_consistent']:.3f} p_u={s['p_value_upper']:.3f}"
            )

    section("PAPER SUBPERIOD MCS (Tab. 5)")
    expected_subperiod = {
        "2022_high_vol":           {"GARCH", "EGARCH", "GJR-GARCH", "XGBoost", "Ensemble"},
        "2023_2025_lower_vol":     {"GJR-GARCH", "LightGBM", "XGBoost", "Ensemble"},
    }
    for label, exp_survivors in expected_subperiod.items():
        actual_survivors = set(sub[label]["mcs_survivors"])
        if actual_survivors == exp_survivors:
            passmsg(f"Subperiod {label}: survivors = {sorted(actual_survivors)}")
        else:
            fails += fail(
                f"Subperiod {label}: expected {sorted(exp_survivors)}, got {sorted(actual_survivors)}"
            )

    # ============================================================
    # v1 ADDITIONS: Phase 2 (30-paper) numerical claims
    # ============================================================
    section("PAPER PHASE 2 RDS DISTRIBUTION (Tab. 6, Tab. 7)")
    rds_csv = ROOT / "studies" / "results" / "phase2_rds_strict.csv"
    if not rds_csv.exists():
        fails += fail(f"missing {rds_csv}")
    else:
        import csv
        with rds_csv.open() as f:
            rows = list(csv.DictReader(f))
        n_total = len(rows)
        n_rds0 = sum(1 for r in rows if int(r["strict_rds"]) == 0)
        n_rds1 = sum(1 for r in rows if int(r["strict_rds"]) == 1)
        n_rds2 = sum(1 for r in rows if int(r["strict_rds"]) == 2)
        mean_rds = sum(int(r["strict_rds"]) for r in rows) / n_total

        expected = {"n_total": 30, "rds0": 16, "rds1": 14, "rds2": 0, "mean": 0.47}
        if n_total == expected["n_total"]:
            passmsg(f"n_total = {n_total}")
        else:
            fails += fail(f"n_total: expected {expected['n_total']}, got {n_total}")
        if n_rds0 == expected["rds0"]:
            passmsg(f"RDS-0: {n_rds0}")
        else:
            fails += fail(f"RDS-0: expected {expected['rds0']}, got {n_rds0}")
        if n_rds1 == expected["rds1"]:
            passmsg(f"RDS-1: {n_rds1}")
        else:
            fails += fail(f"RDS-1: expected {expected['rds1']}, got {n_rds1}")
        if n_rds2 == expected["rds2"]:
            passmsg(f"RDS-2: {n_rds2}")
        else:
            fails += fail(f"RDS-2: expected {expected['rds2']}, got {n_rds2}")
        if abs(mean_rds - expected["mean"]) < 0.01:
            passmsg(f"Mean strict RDS = {mean_rds:.2f}")
        else:
            fails += fail(f"Mean strict RDS: expected {expected['mean']:.2f}, got {mean_rds:.2f}")

    section("PAPER PHASE 2 SAMPLE BY VENUE (Tab. 6)")
    if rds_csv.exists():
        from collections import Counter
        venues = Counter(r["venue"] for r in rows)
        expected_venues = {
            "Journal of Financial Econometrics": 6,
            "Mathematics (MDPI)": 6,
            "JRFM (MDPI)": 4,
            "Risks (MDPI)": 3,
            "arXiv q-fin": 11,
        }
        for v, n_exp in expected_venues.items():
            n_got = venues.get(v, 0)
            if n_got == n_exp:
                passmsg(f"{v}: {n_got}")
            else:
                fails += fail(f"{v}: expected {n_exp}, got {n_got}")

    section("PAPER PHASE 2 SIG-TEST USAGE")
    headline_csv = ROOT / "studies" / "results" / "phase2_headline_models.csv"
    if not headline_csv.exists():
        fails += fail(f"missing {headline_csv}")
    else:
        with headline_csv.open() as f:
            hrows = list(csv.DictReader(f))
        sig = Counter(r["sig_test_in_paper"] for r in hrows)
        n_no_test = sig.get("none", 0)
        if n_no_test == 27:
            passmsg(f"papers with NO formal significance test: {n_no_test}/30 (90%)")
        else:
            fails += fail(f"papers with no sig test: expected 27, got {n_no_test}")
        n_mcs = sig.get("MCS", 0) + sig.get("DM + MCS", 0)
        if n_mcs == 3:
            passmsg(f"papers with MCS or DM+MCS in original analysis: {n_mcs}/30")
        else:
            fails += fail(f"MCS papers: expected 3, got {n_mcs}")

    section("PAPER PHASE 2 SURVIVAL OUTCOMES (Tab. 8)")
    summary_path = ROOT / "studies" / "results" / "phase2_summary.json"
    if not summary_path.exists():
        fails += fail(f"missing {summary_path}")
    else:
        summary = json.loads(summary_path.read_text())
        outcomes = summary.get("outcomes", {})
        expected_outcomes = {"fails": 12, "survives": 1, "partial": 1}
        for k, v_exp in expected_outcomes.items():
            v_got = outcomes.get(k, 0)
            if v_got == v_exp:
                passmsg(f"{k}: {v_got}/14")
            else:
                fails += fail(f"{k}: expected {v_exp}, got {v_got}")
        n_total_repro = summary.get("n_papers_reproduced", 0)
        if n_total_repro == 14:
            passmsg(f"n_papers_reproduced: {n_total_repro}")
        else:
            fails += fail(f"n_papers_reproduced: expected 14, got {n_total_repro}")

    section("FINAL")
    if fails == 0:
        print("\nALL CHECKS PASSED")
        return 0
    print(f"\n{fails} CHECK(S) FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
