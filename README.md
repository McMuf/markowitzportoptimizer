## Why this project

I recently looked into portfolio theory/management and stumbled across Markowitz's mean-variance framework. 
After learning a bit of linear algebra, statistics, and some lagrange multipliers from multivariable calculus,
I decided why not make one from scratch? 
Essentially, Markowitz's framework is minimizing risk (variance) for a given expected return, using the covariance
structure between assets. It's taught everywhere, but rarely stress-tested by the
people implementing it. Even when it is implemented, people tend to use modules like scipy.optimize or PyPortfolioOpt instead of building
it from the ground up using Lagrange multipliers. It also tests whether a portfolio that's "optimal" on historical data actually performs
well going forward and compares a fix (covariance shrinkage) against the naive version and a simple 1/N benchmark.(This is intended to be a learning experience and is by no means a superior way of implementing
Markowitz's mean-variance framework)

## Features

- Closed-form mean-variance optimizer: solves for optimal weights directly
  from the Lagrangian system, not via a numerical solver (like scipy module)
- Efficient frontier construction across a range of target returns (stocks outlined later)
- Tangency (max Sharpe) portfolio is cross-validated against a `scipy.optimize`
  solution for validation
- Ledoit-Wolf covariance shrinkage is to address estimation error in near-singular
  covariance matrices
- Out-of-sample backtest: naive Markowitz vs. shrinkage-adjusted Markowitz vs.
  equal-weight (1/N), evaluated on realized (not estimated) returns
- Test suite validates the closed-form solution against numerical methods and
  checking basic invariants (weights sum to 1, target return is hit)

Here's what the project structure looks like: 
src/
├── data.py         # price data fetching + local caching
├── stats.py         # mu, Sigma computation from historical returns
├── optimizer.py      # closed-form solver + tangency portfolio
├── shrinkage.py       # Ledoit-Wolf covariance + train/test backtest logic
└── plotting.py        # efficient frontier, covariance heatmap, backtest charts
notebooks/
└── analysis.ipynb      # analysis
tests/
└── test_optimizer.py    # correctness checks against scipy, sanity invariants


## Key design decisions 

- np.linalg.solve over np.linalg.inv because  matrix inversion increases
  numerical error when the covariance matrix is near-singular. Therefore solving the linear
  system directly is more stable and accurate. 
- Closed-form implementation validated against scipy.optimize (checking answers)
- Annualized daily returns (×252) is standard convention for comparing to
  typical annual return/risk figures.
- Ledoit-Wolf shrinkage over raw sample covariance is the sample covariance
  matrix is a poor estimator when the number of assets approaches the number of
  observations; shrinkage blends it toward a more stable, lower-variance target.

## Results (will be updated sometimes)

Backtest config: 8-asset universe (AAPL, MSFT, AMZN, GOOGL, JPM, XOM, JNJ, PG),
2015-01-01 to 2024-01-01 daily prices, 60% train / 40% test split, 2% annualized
risk-free rate. Tangency (max-Sharpe) portfolio fit on the training window,
weights held fixed and evaluated on realized test-window returns.

Naive Markowitz out-of-sample Sharpe: `-0.10`
Shrinkage-adjusted Markowitz out-of-sample Sharpe: `-0.08`
Equal-weight (1/N) out-of-sample Sharpe: `1.02`

Both Markowitz variants produced a negative out-of-sample Sharpe ratio, while the
naive 1/N benchmark comfortably outperformed. This is an interesting thing to point out, however it is a well-documented
result. (DeMiguel, Garlappi & Uppal, 2009)
The reason why is mean-variance weights are
so sensitive to estimation error in μ that the "optimal" in-sample portfolio can be
 harmful out-of-sample, and simple heuristics often beat it in practice.
 Another thing is that shrinkage helps at the margin here but doesn't flip the sign. As well, numbers will shift with the universe, date range, and
train/test split (solution is to re-run `notebooks/analysis.ipynb` to reproduce or vary them).

## Limitations

Mean-variance optimization assumes stable, correctly-estimated inputs (μ, Σ).
In practice they are noisy estimates from limited historical data. This project
demonstrates that sensitivity directly rather than negating it (in the best interests of reducing margin of error).

## How to Run it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb
```
