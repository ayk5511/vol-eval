# Studies

Empirical demonstrations and Phase 2 collection scaffolding for the vol-eval companion paper.

## Phase 1: completed (v0 paper)

- `01_paper2_demonstration.py` applies vol-eval's full battery to the seven-model panel of Khan (2026), the volatility-forecasting horse race. Every numerical value in the v0 companion paper traces back to this script's JSON outputs in `results/`.

To rerun:

```bash
cd /path/to/paper3-vol-eval
python studies/01_paper2_demonstration.py
```

This requires Paper 2's forecast panel to be available at the path hard-coded in the script. If Paper 2's repo is checked out elsewhere, edit the `PAPER2_FORECASTS` constant.

## Phase 2: queued (30-paper re-analysis)

The plan documented in `../PHASE_2_AND_3.md` is to apply vol-eval to a sample of 30 published volatility-forecasting comparison papers from 2018-2025, score each on the Reproducibility Disclosure Score rubric, and tabulate what fraction of headline "wins" survive significance testing.

### Workflow when ready

1. **Identify candidate papers.** Search by topic-and-venue criteria documented in `PHASE_2_AND_3.md`. Aim for 30 papers across the target journals + SSRN.

2. **Per-paper data acquisition.** For each paper:
   - Create `per_paper/<paper_id>/`
   - Add a one-line citation in `02_paper_collection_template.csv` extended for that paper
   - Acquire the forecasts:
     - **RDS-2 papers**: clone the original repo, rerun, extract the per-day forecasts of (a) the author-claimed winner, (b) the author-claimed runner-up, (c) any other models in the comparison
     - **RDS-1 papers**: re-implement the headline models from the paper's description; document the re-implementation choices
     - **RDS-0 papers**: skip the significance analysis; record the unreproducibility
   - Write the forecasts as a parquet at `per_paper/<paper_id>/forecasts.parquet` with columns `actual`, `winner`, `runner_up`, plus optional `<other_model_name>` columns

3. **Run vol-eval per paper.** The driver `03_apply_voleval_phase2.py` discovers all `per_paper/` subdirectories and applies DM / MCS / SPA to each, writing `voleval_result.json` per paper.

4. **Aggregate.** The same script aggregates per-paper results into `results/phase2_summary.json` with the headline "fraction of wins surviving significance" tabulation.

5. **Write up.** The Phase 3 paper (companion to vol-eval, follow-on to the v0) presents the aggregate findings.

### Forecasts.parquet schema

| Column | Required | Type | Notes |
|---|---|---|---|
| `actual` | yes | float64 | Realized volatility proxy used by the paper |
| `winner` | yes | float64 | Author-claimed best forecaster's per-day forecast |
| `runner_up` | yes | float64 | Author-claimed runner-up's per-day forecast |
| `<other_model>` | no | float64 | Other models in the paper's comparison (one column each) |

The index should be a date or sequential integer. NaN values are dropped before testing.

### Time budget per paper

- RDS-2 paper: ~2 hr (clone, install, rerun, extract forecasts)
- RDS-1 paper: ~6-8 hr (re-implementation)
- RDS-0 paper: ~30 min (document unreproducibility)

For 30 papers with the typical RDS distribution from Khan (2026) (mean RDS = 0.78), expect ~120-180 hours of total work. This is the multi-week project that produces the empirical Phase 3 paper.
