# Markowitz Portfolio Optimization: Does It Survive Out-of-Sample?

A from-scratch implementation of mean-variance portfolio optimization — closed-form
derivation, not a library call — extended to test whether the "optimal" portfolio
it produces actually holds up on data it hasn't seen.

## Why this project

Markowitz's mean-variance framework is the foundation of modern portfolio theory:
minimize risk (variance) for a given expected return, using the covariance
structure between assets. It's taught everywhere, but rarely stress-tested by the
people implementing it. This project does two things most tutorial versions skip:

1. Derives and implements the closed-form solution by hand (via Lagrange
   multipliers), rather than calling `scipy.optimize` or `PyPortfolioOpt` directly.
2. Tests whether a portfolio that's "optimal" on historical data actually performs
   well going forward — and compares a fix (covariance shrinkage) against the naive
   version and a simple 1/N benchmark.

## Features

- **Closed-form mean-variance optimizer** — solves for optimal weights directly
  from the Lagrangian system, not via a numerical solver
- **Efficient frontier** construction across a range of target returns
- **Tangency (max Sharpe) portfolio**, cross-validated against a `scipy.optimize`
  solution to confirm correctness
- **Ledoit-Wolf covariance shrinkage** to address estimation error in near-singular
  covariance matrices
- **Out-of-sample backtest**: naive Markowitz vs. shrinkage-adjusted Markowitz vs.
  equal-weight (1/N), evaluated on realized (not estimated) returns
- **Test suite** validating the closed-form solution against numerical methods and
  checking basic invariants (weights sum to 1, target return is hit)

## Project structure

```
src/
├── data.py         # price data fetching + local caching
├── stats.py         # mu, Sigma computation from historical returns
├── optimizer.py      # closed-form solver + tangency portfolio
├── shrinkage.py       # Ledoit-Wolf covariance + train/test backtest logic
└── plotting.py        # efficient frontier, covariance heatmap, backtest charts
notebooks/
└── analysis.ipynb      # full narrative: theory → derivation → results → limitations
tests/
└── test_optimizer.py    # correctness checks against scipy, sanity invariants
```

## Key design decisions (and why)

- **`np.linalg.solve` over `np.linalg.inv`** — explicit matrix inversion amplifies
  numerical error when the covariance matrix is near-singular; solving the linear
  system directly is more stable and is standard practice.
- **Closed-form implementation validated against `scipy.optimize`** — not used as
  the primary solver, but as a correctness check on the hand-derived math.
- **Annualized daily returns (×252)** — standard convention for comparing to
  typical annual return/risk figures.
- **Ledoit-Wolf shrinkage over raw sample covariance** — the sample covariance
  matrix is a poor estimator when the number of assets approaches the number of
  observations; shrinkage blends it toward a more stable, lower-variance target.

## Results

Backtest config: 8-asset universe (AAPL, MSFT, AMZN, GOOGL, JPM, XOM, JNJ, PG),
2015-01-01 to 2024-01-01 daily prices, 60% train / 40% test split, 2% annualized
risk-free rate. Tangency (max-Sharpe) portfolio fit on the training window,
weights held fixed and evaluated on realized test-window returns.

- Naive Markowitz out-of-sample Sharpe: `-0.10`
- Shrinkage-adjusted Markowitz out-of-sample Sharpe: `-0.08`
- Equal-weight (1/N) out-of-sample Sharpe: `1.02`

Both Markowitz variants produced a *negative* out-of-sample Sharpe ratio, while the
naive 1/N benchmark comfortably outperformed. This is a striking but well-documented
result in the literature (DeMiguel, Garlappi & Uppal, 2009): mean-variance weights are
so sensitive to estimation error in μ that the "optimal" in-sample portfolio can be
actively harmful out-of-sample, and simple heuristics often beat it in practice.
Shrinkage helps at the margin here but doesn't flip the sign — consistent with the
Limitations section below. Numbers will shift with the universe, date range, and
train/test split; re-run `notebooks/analysis.ipynb` to reproduce or vary them.

## Limitations

Mean-variance optimization assumes stable, correctly-estimated inputs (μ, Σ) —
in practice both are noisy estimates from limited historical data, and the
"optimal" portfolio can be highly sensitive to estimation error. This project
demonstrates that sensitivity directly rather than assuming it away, and shows
one standard practitioner mitigation (shrinkage) — though even shrinkage doesn't
fully close the gap to realized performance.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb
```
