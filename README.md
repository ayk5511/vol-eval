# vol-eval

Volatility forecast evaluation: standard loss functions plus Diebold-Mariano,
Model Confidence Set, SPA, and Reality Check tests.

A small, opinionated Python package. Built for the version of financial
machine learning that takes evaluation discipline seriously: every pairwise
comparison reports a significance test, every multi-model comparison reports
a confidence set, every paper ships with the audit script that produced the
numbers in its tables.

## Why this package

Most empirical volatility-forecasting papers report a full-sample QLIKE (or
RMSE) ranking and stop. The reader has no way to tell whether the headline
"win" is statistically distinguishable from the runner-up, or whether the
ranking survives a regime decomposition. The Diebold-Mariano test is ten
lines of Python, the Model Confidence Set is a few hundred, and there is no
good reason every published comparison should not run them.

`vol-eval` packages the standard tests behind a clean Python API so that any
reasonably-equipped researcher can apply them in a few minutes. The companion
paper (in production) re-analyzes 30 recent published volatility-forecasting
comparisons and reports what fraction of "wins" survive significance testing.

## Install

```bash
pip install vol-eval
```

Requirements: Python 3.10+, NumPy 1.24+, SciPy 1.11+, Pandas 2.0+.

## Quick start

```python
import numpy as np
from vol_eval import qlike, mse, mz_r2, dm_test, model_confidence_set, spa_test

# Three forecasters of realized volatility
actual = ...           # shape (T,) realized volatility
forecast_garch = ...   # shape (T,)
forecast_lgbm = ...    # shape (T,)
forecast_har = ...     # shape (T,)

# Pointwise loss functions
qlike(actual, forecast_garch)   # 0.345
mse(actual, forecast_garch)     # 0.0054
mz_r2(actual, forecast_garch)   # 0.38

# Diebold-Mariano: is forecast_garch significantly better than forecast_lgbm?
result = dm_test(actual, forecast_garch, forecast_lgbm, loss="qlike", h=5)
print(result)
# DMResult(loss='qlike', h=5, n=980, mean_diff=-0.018, t=-2.14, p=0.033, winner='a')

# Model Confidence Set: which models survive at the 90% confidence level?
mcs = model_confidence_set(
    actual,
    {"garch": forecast_garch, "lgbm": forecast_lgbm, "har": forecast_har},
    loss="qlike",
    alpha=0.10,
    n_bootstrap=1000,
)
print(mcs.survivors)   # ['garch', 'lgbm']
print(mcs.eliminated)  # ['har']

# Hansen SPA: is forecast_garch superior to all competitors?
spa = spa_test(
    actual,
    benchmark=forecast_garch,
    competitors={"lgbm": forecast_lgbm, "har": forecast_har},
    loss="qlike",
)
print(spa.p_value_consistent)  # 0.42 -> cannot reject H0; garch is fine
```

## What's included

### Loss functions (`vol_eval.losses`)

| Function | Reference | Notes |
|---|---|---|
| `qlike(actual, forecast)` | Patton (2011) | Robust to noise in volatility proxy. The right default for volatility forecasting. |
| `mse(actual, forecast)` | Standard | |
| `mae(actual, forecast)` | Standard | |
| `rmse(actual, forecast)` | Standard | Square root of MSE |
| `mz_r2(actual, forecast)` | Mincer-Zarnowitz (1969) | Higher is better, unlike all the others |

### Significance tests (`vol_eval.tests`)

| Function | Reference | What it answers |
|---|---|---|
| `dm_test(actual, fa, fb, ...)` | Diebold-Mariano (1995) | Are two forecasters significantly different on average loss? |
| `model_confidence_set(actual, forecasts, ...)` | Hansen, Lunde, Nason (2011) | Which subset of models is statistically equivalent to the best? |
| `spa_test(actual, benchmark, competitors, ...)` | Hansen (2005) | Is the benchmark not inferior to any competitor? Returns lower / consistent / upper p-values. |
| `reality_check(actual, benchmark, competitors, ...)` | White (2000) | Conservative variant of SPA. Equivalent to SPA's upper-bound p-value. |

All significance tests use a Newey-West HAC standard error with bandwidth
`h - 1` (rule of thumb for an h-step-ahead forecast), and the bootstrap-based
tests use Politis-Romano stationary bootstrap.

## Design choices

- **Single-purpose**. The package does forecast evaluation. It does not fit
  models, manage data pipelines, or provide a CLI. It is designed to be
  imported by your existing workflow.
- **Functional API**. Every test takes arrays and returns a typed result
  object. No state, no side effects.
- **Reproducibility-friendly**. Every test that uses random sampling accepts
  an optional `numpy.random.Generator` for reproducible runs.
- **Honest defaults**. QLIKE is the default loss. Bootstrap tests default to
  1000 replications and stationary-bootstrap block size scaled with sample
  size. These are the choices the literature actually supports.

## Companion paper

This package is the technical foundation for:

> Khan, A. (2026). *Are Published Volatility Forecasting "Wins" Statistically
> Significant? A vol-eval Re-Analysis of 30 Recent Papers (2018–2025).*
> Working Paper, in production.

The paper applies `vol-eval` to 30 published comparison papers from 2018-2025
and reports the fraction of "wins" that survive Diebold-Mariano testing,
fall outside the Model Confidence Set, or fail Hansen SPA. Spoiler from the
pilot run: a substantial minority do not.

## Citation

If you use `vol-eval` in research, please cite:

```bibtex
@software{KhanVolEval2026,
  title  = {vol-eval: A Python Package for Volatility Forecast Evaluation},
  author = {Khan, Akram},
  year   = {2026},
  url    = {https://github.com/ayk5511/vol-eval},
  version = {0.1.0}
}
```

GitHub renders a one-click "Cite this repository" button via `CITATION.cff`.

## License

MIT. See [LICENSE](LICENSE).

## Contributing

Issues and pull requests welcome. Particularly interested in:

- Implementations of additional tests (e.g., the Hansen-Lunde-Nason MCS with
  alternative variance estimators)
- Worked examples on different asset classes / forecast horizons
- Bug reports with reproduction code

The development install:

```bash
git clone https://github.com/ayk5511/vol-eval
cd vol-eval
pip install -e ".[dev]"
pytest
```
