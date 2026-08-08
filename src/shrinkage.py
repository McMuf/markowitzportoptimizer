# Ledoit-Wolf covariance + train/test backtest logic (see README "Project structure")


from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from . import optimizer, stats


def shrunk_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Annualized Ledoit-Wolf shrinkage covariance estimate."""
    # The Ledoit-Wolf shrinkage over raw sample covariance is because the sample covariance matrix is a poor estimator
    # when the number of assets approaches the number of observations
    lw = LedoitWolf().fit(returns.to_numpy())
    return lw.covariance_ * stats.TRADING_DAYS


@dataclass
class BacktestResult:
    naive_sharpe: float
    shrinkage_sharpe: float
    equal_weight_sharpe: float
    naive_weights: np.ndarray
    shrinkage_weights: np.ndarray
    naive_realized_returns: pd.Series
    shrinkage_realized_returns: pd.Series
    equal_weight_realized_returns: pd.Series


def _realized_stats(weights: np.ndarray, test_returns: pd.DataFrame, rf: float = 0.0) -> tuple[float, pd.Series]:
  
    port_returns = test_returns.to_numpy() @ weights
    port_returns = pd.Series(port_returns, index=test_returns.index)
    ann_return = port_returns.mean() * stats.TRADING_DAYS
    ann_vol = port_returns.std() * np.sqrt(stats.TRADING_DAYS)
    sharpe = (ann_return - rf) / ann_vol
    return sharpe, port_returns


def train_test_backtest(
    returns: pd.DataFrame,
    train_frac: float = 0.6,
    rf: float = 0.0,
) -> BacktestResult:
    """Fit tangency portfolios on the training window, evaluate on the
    (unseen) test window.

    Compares three portfolios, all evaluated out-of-sample:
      - naive Markowitz: tangency portfolio from raw sample covariance
      - shrinkage Markowitz: tangency portfolio from Ledoit-Wolf covariance
      - equal-weight (1/N): no estimation at all, the standard benchmark
    """
   
    split = int(len(returns) * train_frac)
    train, test = returns.iloc[:split], returns.iloc[split:]
    if len(test) == 0:
        raise ValueError("train_frac too high: no data left for the test window")

    mu_train, sigma_naive = stats.mu_sigma(train)
    sigma_shrunk = shrunk_covariance(train)


    w_naive = optimizer.tangency_portfolio(mu_train, sigma_naive, rf=rf)
    w_shrunk = optimizer.tangency_portfolio(mu_train, sigma_shrunk, rf=rf)
    w_equal = np.ones(len(mu_train)) / len(mu_train)

    naive_sharpe, naive_ret = _realized_stats(w_naive, test, rf=rf)
    shrink_sharpe, shrink_ret = _realized_stats(w_shrunk, test, rf=rf)
    equal_sharpe, equal_ret = _realized_stats(w_equal, test, rf=rf)

    return BacktestResult(
        naive_sharpe=naive_sharpe,
        shrinkage_sharpe=shrink_sharpe,
        equal_weight_sharpe=equal_sharpe,
        naive_weights=w_naive,
        shrinkage_weights=w_shrunk,
        naive_realized_returns=naive_ret,
        shrinkage_realized_returns=shrink_ret,
        equal_weight_realized_returns=equal_ret,
    )
