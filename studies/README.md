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

## Phase 2: in progress (30-paper re-analysis)

The plan documented in `../PHASE_2_AND_3.md` is to apply vol-eval to a sample of 30 published volatility-forecasting comparison papers from 2018-2025, score each on the Reproducibility Disclosure Score rubric, and tabulate what fraction of headline "wins" survive significance testing.

### Pipeline scripts (in execution order)

| Script | Purpose | Inputs | Outputs |
|---|---|---|---|
| `04_search_candidates.py` | Bibliographic search via Crossref + OpenAlex + arXiv | (none, hard-coded venues) | `results/candidates_raw.csv`, `results/candidates_filtered.csv` |
| `05_narrow_candidates.py` | Strict horse-race filter + per-venue stratified caps | `candidates_filtered.csv` | `results/candidates_top.csv` |
| (manual) | Final selection of 30 papers from `candidates_top.csv` | `candidates_top.csv` | `candidate_30.csv` |
| `06_check_unpaywall.py` | Find legal free copies via Unpaywall API | `candidate_30.csv` | `candidate_30_access.csv` |
| `07_download_papers.py` | Download free PDFs (arXiv + Unpaywall + MDPI via curl) | `candidate_30_access.csv` | `per_paper/<paper_id>/source.pdf` |
| `08_audit_phase2_setup.py` | Verify integrity of the 30-paper setup | all of the above | console report |
| `03_apply_voleval_phase2.py` | Apply DM/MCS/SPA per paper + aggregate | `per_paper/<paper_id>/forecasts.parquet` | per-paper `voleval_result.json` + `results/phase2_summary.json` |

### Re-acquiring the 30 PDFs

Source PDFs are gitignored (50 MB combined). To rebuild from a fresh clone:

```bash
python3 studies/06_check_unpaywall.py    # produces candidate_30_access.csv
python3 studies/07_download_papers.py    # downloads 17 of 30 automatically
```

The 13 MDPI papers (`P03`, `P08`-`P19`) require a browser download because MDPI rate-limits programmatic access. Open each DOI from `candidate_30.csv` in a browser, click Download PDF, save as `studies/per_paper/<paper_id>/source.pdf`.

After both, run `python3 studies/08_audit_phase2_setup.py` — should print `ALL CHECKS PASSED`.

### Per-paper procedure

For each paper (after PDFs are in place):

1. **RDS scoring**: score 0/1/2 on the [Khan2026Survey](https://ssrn.com/abstract=6562398) rubric.
2. **Headline winner / runner-up extraction**: from the paper's abstract and conclusion.
3. **Reproduce or document**:
   - **RDS-2**: clone the original repo, rerun, extract per-day forecasts of (a) author-claimed winner, (b) runner-up, (c) any other models
   - **RDS-1**: re-implement the headline models from the paper's description; document choices
   - **RDS-0**: skip significance analysis; record the unreproducibility in the CSV
4. Write forecasts as `per_paper/<paper_id>/forecasts.parquet` with columns `actual`, `winner`, `runner_up`, plus optional `<other_model>` columns.
5. **Run** `python3 studies/03_apply_voleval_phase2.py` to apply DM/MCS/SPA per paper and aggregate into `results/phase2_summary.json`.
6. **Write up** the Phase 3 paper (companion to vol-eval, follow-on to the v0) using the aggregate findings.

### Sample frame as of 2026-04-28

| Venue | Papers |
|---|---|
| Journal of Financial Econometrics | 6 |
| Mathematics (MDPI) | 6 |
| Journal of Risk and Financial Management (MDPI) | 4 |
| Risks (MDPI) | 3 |
| arXiv q-fin | 11 |
| **Total** | **30** |

Originally-targeted Wiley/Elsevier journals (JoF, IJF, JEF, QF, JFDS) are absent due to PDF access constraints during programmatic download. This sample-frame deviation is documented in `../PHASE_2_AND_3.md` and will be disclosed as a limitation in the eventual Phase 2 paper.

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
