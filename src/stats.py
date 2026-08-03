

# fyi: this is the naive/raw estimator on purpose. shrinkage.py has the
# Ledoit-Wolf version of Sigma, and the whole point of the project is comparing this raw version against that one out-of-sample....

from __future__ import annotations

import numpy as np
import pandas as pd


# 252 = approx number of US trading days in a year.
TRADING_DAYS = 252 


def annualized_mean(returns: pd.DataFrame) -> np.ndarray:
    """Annualized expected return per asset, as a 1D numpy array."""
    #daily mean return * 252, not (1+r)^252 - 1. this is the simple/ linear annualization, consistent with the simple (not log) returns data.py produces.
    return returns.mean().to_numpy() * TRADING_DAYS


def annualized_cov(returns: pd.DataFrame) -> np.ndarray:
    """Annualized sample covariance matrix, as a 2D numpy array."""
    # variance scales linearly with time under the iid-returns daily cov * 252 (not *sqrt(252), that's for vol/std).
    return returns.cov().to_numpy() * TRADING_DAYS


def mu_sigma(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: (mu, Sigma) both annualized from daily returns."""
    # reminder: optimizer.py wants plain numpy arrays (mu, Sigma), not pandas
  
    return annualized_mean(returns), annualized_cov(returns)
