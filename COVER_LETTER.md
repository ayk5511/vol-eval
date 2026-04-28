# Cover letter — vol-eval paper

This document holds the cover letter and SSRN submission metadata. Two
versions: a short SSRN-form abstract (for the SSRN upload form) and a
longer journal cover letter (for *Journal of Financial Data Science* or
*International Journal of Forecasting* future submission).

---

## SSRN submission metadata

**Title:** vol-eval: A Python Package for Volatility Forecast Evaluation, with a Re-Analysis of 30 Published Comparison Papers (2018-2025)

**Author:** Akram Khan (Independent Researcher)
**ORCID:** 0009-0002-7521-8648
**Affiliation field:** Independent Researcher

**Keywords:** volatility forecasting, Diebold-Mariano test, Model Confidence Set, Hansen SPA, Reality Check, QLIKE, forecast evaluation, reproducibility, open-source software, replication

**JEL codes:** C12, C15, C22, C52, C53, G17

**SSRN networks (suggested):**
- Econometric & Statistical Methods - Special Topics
- Risk Management
- Financial Engineering
- Quantitative Finance & Statistics

**Distribution:** Yes (free, immediate)
**Public:** Yes
**License:** CC BY 4.0 (so that the package's MIT license and the paper's open-license align)

**Abstract** (~5,000 chars limit, the paper's PDF abstract is reproduced verbatim):

> We introduce vol-eval, an open-source Python package that implements the canonical battery of significance tests for volatility forecast evaluation: QLIKE, MSE, MAE, Mincer-Zarnowitz R^2, Diebold-Mariano (1995) with Newey-West HAC standard errors, the Hansen-Lunde-Nason (2011) Model Confidence Set with stationary bootstrap, Hansen's (2005) SPA test with three p-value variants, and White's (2000) Reality Check. The package follows a functional API with typed result objects, ships under MIT license, and is verified by 34 unit tests including cross-validation against the arch package's reference implementation.
>
> We apply vol-eval both to a self-contained seven-model horse race on S&P 500 realized volatility (the panel of Khan 2026) and to a sample of 30 published volatility-forecasting comparison papers from 2018-2025, drawn from the Journal of Financial Econometrics, three open-access MDPI venues (Mathematics, Risks, Journal of Risk and Financial Management), and arXiv q-fin.
>
> Three layered findings emerge. First, a disclosure crisis: the strict-reading mean Reproducibility Disclosure Score across the 30-paper sample is 0.47, materially lower than the 0.78 reported by Khan (2026) for the broader 99-paper ML-finance sample; zero of the 30 papers ship both code and publicly accessible data. Second, a methodological gap: 27 of 30 papers (90%) declare a headline winner without applying any formal predictive-accuracy test (DM, MCS, SPA, or Reality Check). Third, a robustness gap: in the 14 papers we could reproduce (the strict-RDS-1 subset, where data is publicly accessible), 12 (86%) of paper-claimed wins do not survive significance testing under the vol-eval battery; only one paper has its claim survive in full, with one additional paper showing partial survival on a subset of assets. We document each per-paper reproduction and its caveats in detail.
>
> We release the full vol-eval source, the per-paper reproductions, and an audit script that re-derives every numerical claim in this paper at github.com/ayk5511/vol-eval, scoring 2 on the Reproducibility Disclosure Score rubric of Khan (2026).

---

## Journal cover letter (for future submission to JFDS / IJF / similar)

> *To the Editors:*
>
> I am pleased to submit the manuscript "vol-eval: A Python Package for Volatility Forecast Evaluation, with a Re-Analysis of 30 Published Comparison Papers (2018-2025)" for consideration. The paper combines a methodological contribution (an open-source Python package implementing the canonical Diebold-Mariano, Model Confidence Set, SPA, and Reality Check tests for volatility forecast evaluation) with an empirical contribution (a re-analysis of 30 published comparison papers from 2018-2025 testing whether their headline "wins" survive formal significance testing).
>
> The empirical findings are, I believe, of interest to the journal's readership for three reasons.
>
> *First, the disclosure pattern is sharper than previously documented in this subfield.* Of 30 published volatility-forecasting comparison papers spanning 2018-2025, none ship both code and publicly accessible data together. The strict-reading mean Reproducibility Disclosure Score (per the rubric of Khan 2026) is 0.47, materially lower than the 0.78 reported in the broader 99-paper ML-finance sample. The pattern is venue-driven: the Journal of Financial Econometrics cohort uniformly relies on paywalled high-frequency tick data (NYSE TAQ, Refinitiv Datastream, Thomson Reuters), while the open-access MDPI cohort and arXiv preprints disclose public sources but do not ship code.
>
> *Second, the methodological pattern is striking.* Of the 30 papers, 27 (90%) declare a headline "winner" model without applying any formal predictive-accuracy test. Only three papers report a Diebold-Mariano or Model Confidence Set result. The remaining 27 declare winners on the basis of point-estimate ranking by MSE / RMSE / MAE / R^2 / QLIKE, sometimes with claimed effect sizes as large as "39-95% RMSE reduction" but with no confidence intervals and no formal hypothesis test.
>
> *Third, the empirical robustness pattern, with appropriate caveats, is the strongest of the three.* For the 14 papers in our sample whose data is publicly accessible (the strict-RDS-1 subset), we re-implemented the headline winner and runner-up models on the disclosed data source and applied the full vol-eval battery. In 12 of 14 (86%), the paper's claimed win does not survive: either the claimed winner is not the lowest-QLIKE model in our reproduction, or the Diebold-Mariano test fails to reject equal predictive accuracy at 5%. Only one paper had its claim survive in full; one additional paper showed partial survival on a subset of cryptocurrencies.
>
> The reproductions are necessarily approximations of the originals because the RDS-1 papers ship no code; we re-implement headline architectures from each paper's prose description and document our hyperparameter choices in per-paper reproduction logs. Three reproductions involved substantial library or feature substitutions that we explicitly flag in Section 5.5 of the manuscript and in the failure tabulation. We have been careful not to overclaim from this 14-paper subset; the manuscript's discussion section enumerates five caveats that limit the strength of the headline 86% number.
>
> The work also provides infrastructure for the field to do better. The vol-eval package is MIT-licensed at github.com/ayk5511/vol-eval, has 34 unit tests including cross-validation against the canonical arch package, and implements each test as a one-line API call. The paper itself ships with an audit script that re-derives every numerical claim from JSON outputs, providing a mechanical guard against transcription errors of the kind we have seen in related work.
>
> The manuscript has not been submitted elsewhere and is not under consideration at any other journal. I have no competing interests to declare. All data and code are publicly available; the 30 source PDFs are in the project repository's reproduction directories with re-acquisition URLs documented for reproducibility.
>
> Thank you for your consideration.
>
> Sincerely,
>
> Akram Khan
> Independent Researcher
> ORCID: 0009-0002-7521-8648
> Email: 1819ak@gmail.com

---

## Suggested reviewer pool (for journal submission, not SSRN)

When journals ask for suggested reviewers, the following are plausible:

- **Methodological / forecast-evaluation authors:**
  - Peter Reinhard Hansen (Aarhus, author of the SPA test and MCS)
  - Andrew Patton (Duke, author of QLIKE and the imperfect-vol-proxy paper)
  - Bezirgen Veliyev (Aarhus, co-author of P01 Christensen-Siggaard-Veliyev 2023)

- **vol-forecasting subfield, both econometric and ML:**
  - Marcos Lopez de Prado (ADIA Lab, machine learning in finance)
  - Robert Engle (NYU, GARCH originator)
  - Andrea Bucci (Macerata, single-author of P02 in our sample)

- **Reproducibility / replication:**
  - Anna Dreber (Stockholm, social-science replication crises)
  - Brian Nosek (Center for Open Science)

These should be specified as suggested reviewers only if the journal's submission system supports it; some journals do not.

---

## Anticipated reviewer concerns and responses

A few likely reviewer concerns are worth pre-addressing:

**Q: "Your reproductions are not the paper's exact code. How do we know your 86% finding isn't just a result of your hyperparameter choices?"**

A: Honest framing. We use standard hyperparameter defaults documented in each paper's reproduction_log.json. Where the paper specifies differently, we follow the paper. Section 5.5 walks through five caveats. The 86% figure is robust to dropping the three most heavily-caveated papers (P11, P12, P24) — without those, the failure rate becomes 9 of 11 (82%). The pattern, not the precise number, is the contribution.

**Q: "JFEcon's high-quality econometric papers are absent from the reproducibility analysis. Doesn't that bias your finding?"**

A: Yes, and we say so explicitly (Section 5.5, Limitation 4). The 14-paper reproducible subset is correlated with venue. The 86% figure is the rate among the openly-reproducible subset, not the rate one would expect across the full 30-paper population. This is a real limitation; we do not claim the 86% generalises to JFEcon.

**Q: "Why apply the Reproducibility Disclosure Score this way? The strict reading is harsh."**

A: The strict reading aligns with the original rubric definition in Khan (2026): RDS-2 requires both code and openly-accessible data. Lenient readings (counting commercial-but-disclosed sources as 'accessible') would push the mean RDS up by ~0.2 but would also dilute the rubric's signal. We document the strict reading explicitly so readers can recompute under their preferred reading.

**Q: "What about Hansen SPA's bandwidth and bootstrap choices? Different defaults give different results."**

A: vol-eval uses the standard Newey-West HAC with horizon-adaptive bandwidth and the Politis-Romano stationary bootstrap with a 1/h block-length parameter. These match the conventions in the arch package, against which vol-eval is cross-validated (Section 2.3). The package does not foreclose alternatives; users can pass their own bootstrap design as an argument.

---

## SSRN submission steps (operational checklist)

1. Log in to ssrn.com with `1819ak@gmail.com` / Author ID 11116668.
2. "Submit a Paper" → "Submit New Paper".
3. Upload `paper/submission-ssrn/Khan_2026_vol_eval.pdf`.
4. Title field: paste full title from above.
5. Abstract field: paste the abstract above (under 5000 chars).
6. Author Information: confirm your details are pre-filled.
7. Keywords: paste from above.
8. JEL Codes: paste from above.
9. Networks: select the four listed (or as many as SSRN allows).
10. Distribution: Public, Free, Immediate.
11. Verify the upload preview matches the PDF before final submit.
12. Wait 1-3 business days for SSRN approval.
13. Once approved: link from your SSRN profile, post to LinkedIn, file to ORCID.

---

## After SSRN: post-submission checklist

- [ ] SSRN approval email received → save screenshot to `evidence/`
- [ ] Add Paper 3 to your SSRN author profile
- [ ] Update website-next `/research` route to include the new paper
- [ ] Update STATUS.md and EVIDENCE_TRACKER.md with the new SSRN abstract ID
- [ ] LinkedIn post announcing the paper (one paragraph + abstract link)
- [ ] Email 3-5 academic contacts who work on vol forecasting (Hansen, Patton, Veliyev) with a "you may find this relevant" note + the SSRN URL
- [ ] Submit to *Journal of Financial Data Science* or *International Journal of Forecasting* if pursuing peer review track
- [ ] Optional: arXiv submission (q-fin.ST) — requires endorsement; defer unless endorsement is in hand

This document is committed to the repository as a record. It is gitignored from the SSRN PDF distribution.
