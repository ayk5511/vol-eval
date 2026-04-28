# Phase 2 and 3 plan for Paper 3

This document tracks what comes after the v0.1.0 package release. Phase 1 (the
package itself) is done; Phases 2 and 3 are the empirical study and the
companion paper.

## Phase 1 — package v0.1.0 (DONE, 2026-04-27)

- Loss functions (QLIKE, MSE, MAE, RMSE, MZ-R²)
- Diebold-Mariano with HAC SE
- Hansen-Lunde-Nason Model Confidence Set with stationary bootstrap
- Hansen SPA with three p-value variants (lower, consistent, upper)
- White Reality Check (equivalent to SPA upper-bound)
- 28 unit tests, all passing
- Quickstart example
- README with usage examples
- CITATION.cff for one-click GitHub citation
- MIT license

## Phase 2 — empirical study (target: complete by mid-June 2026)

The study applies vol-eval to a sample of recent published volatility-
forecasting comparison papers and reports what fraction of "wins" survive
significance testing.

### Sample selection

Target: 30 comparison papers from 2018-2025, drawn from:

- Journals: *Journal of Financial Econometrics*, *Journal of Empirical
  Finance*, *Quantitative Finance*, *Journal of Forecasting*, *International
  Journal of Forecasting*, *Journal of Financial Data Science*
- Conferences: ICAIF main track, NeurIPS workshop on Robust ML in Quantitative
  Finance, ICML application track
- SSRN working papers with 50+ downloads in the topic area

Inclusion criteria:
1. Published between 2018-01-01 and 2025-12-31
2. Reports a head-to-head comparison of at least 3 volatility forecasters
3. Has a clearly identified "winner" in the abstract or conclusion
4. Either uses public data or ships code/data sufficient to reproduce

### Per-paper procedure

For each paper:

1. **RDS scoring**: apply the [Reproducibility Disclosure Score](https://ssrn.com/abstract=6562398) rubric.
2. **Reproduce or note non-reproducibility**:
   - RDS-2: clone the code, rerun, get forecast time series for each model
   - RDS-1 (public data, no code): re-implement headline models from paper
     description, get our own forecast series
   - RDS-0 (vendor data, no code): note unreproducibility, skip the
     significance analysis but include in the disclosure tabulation
3. **Apply vol-eval**: for reproducible papers, run DM, MCS, SPA on the
   author-claimed pairwise winner and the runner-up
4. **Tabulate**:
   - Author-claimed winner
   - DM p-value vs. author-claimed runner-up
   - MCS survival status of author-claimed winner
   - SPA p-value treating author's winner as benchmark vs. competitors

### Files to ship

- `studies/01_collect_papers.py`: PRISMA-style collection script with search
  queries against Google Scholar / SSRN / OpenReview APIs
- `studies/paper_database.csv`: structured database of the 30 papers (title,
  authors, year, venue, RDS, reproduced Y/N, headline winner, etc.)
- `studies/02_per_paper_analysis/`: one subdirectory per paper, each with its
  own forecast extraction / re-implementation script and a `result.json`
- `studies/03_aggregate.py`: combines per-paper results into headline tables
- `studies/results/`: aggregate JSON + CSV outputs

### Predicted finding (to be tested, not assumed)

I expect that of the papers we can analyze with significance tests:

- 20-40% of headline DM "wins" are not statistically significant at 5%
- 30-50% of headline winners survive MCS at the 90% confidence level (i.e.,
  most "winners" are tied with at least one runner-up)
- The reproducibility-significance gap is larger for ML-vs-classical
  comparisons than for ML-vs-ML

## Phase 3 — companion paper (target: SSRN by mid-July 2026)

Working title: *Are Published Volatility Forecasting "Wins" Statistically
Significant? A vol-eval Re-Analysis of 30 Recent Papers (2018–2025).*

### Section structure

1. **Introduction** (~700 words)
   - The single-number-reporting habit in financial ML
   - The argument that DM/MCS/SPA should be standard
   - Preview of headline finding
   - Roadmap

2. **The vol-eval package** (~600 words)
   - One-page tour of the API
   - Design choices (functional, no state, reproducibility-friendly)
   - Reference to the GitHub repository

3. **Sample and methodology** (~800 words)
   - PRISMA-style flow for the 30 papers
   - RDS scoring per paper
   - DM/MCS/SPA application protocol
   - Limitations of re-implementation for RDS-1 papers

4. **Headline findings** (~1500 words)
   - Distribution of RDS scores in the sample
   - Distribution of DM p-values for headline pairs
   - MCS survival rates
   - SPA p-values
   - Cross-cuts: by year, by venue, by ML-vs-classical

5. **Case studies** (~1500 words)
   - 3-5 papers analyzed in depth
   - One where the headline survives all tests strongly
   - One where the headline fails DM
   - One where MCS keeps multiple competitors

6. **Discussion** (~1000 words)
   - What this means for the field
   - Recommended reporting standards going forward
   - The case for vol-eval as a default

7. **Reproducibility** (~300 words)
   - This paper's own RDS = 2
   - Repository link, audit script reference

### Companion artifacts

- `paper/main.tex` and standard structure (matching Papers 1 and 2)
- `paper/audit.py`: re-derives every numerical claim from the JSON outputs
- `paper/submission-ssrn/`: PDF + SSRN submission metadata file

## Risks

- **Data collection bottleneck**: getting reproducible code/data for 30 papers
  is the longest single task. Budget 4 weeks for this.
- **Reproduction failures**: some RDS-1 papers may turn out to be
  un-reimplementable in practice. Need a tolerance for skipping cases and
  reporting the skip rate.
- **Author response**: some authors may push back. The framing should be "the
  field has a measurement gap, here is a tool that closes it" — never "this
  specific author is wrong."

## EB1A connection

This paper extends both Paper 1 (which introduced the RDS rubric) and Paper 2
(which demonstrated DM-discipline in a single case study). Together, the
three papers anchor the regulator-defensible-AI thesis on the *evaluation*
side; Paper 4 (audit-trail tooling) will anchor it on the *deployment* side.

The vol-eval package is the durable citation generator: every researcher
running a forecast comparison who uses the package cites the paper. Tools
have longer citation half-lives than empirical findings.
