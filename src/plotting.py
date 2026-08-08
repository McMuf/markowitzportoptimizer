# efficient frontier, covariance heatmap, backtest charts (see README "Project structure")

# reminder -> this file is just visualization, no math happens here btw

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .optimizer import FrontierPoint


def plot_efficient_frontier(
    frontier: list[FrontierPoint],
    mu: np.ndarray,
    Sigma: np.ndarray,
    tickers: list[str] | None = None,
    tangency_weights: np.ndarray | None = None,
    rf: float = 0.0,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    # the frontier curve itself, from optimizer.efficient_frontier()
   
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    risks = [p.risk for p in frontier]
    returns = [p.target_return for p in frontier]
    ax.plot(risks, returns, label="Efficient frontier", color="tab:blue")


    asset_vols = np.sqrt(np.diag(Sigma))
    ax.scatter(asset_vols, mu, color="tab:gray", zorder=3, label="Individual assets")
    if tickers is not None:
        for i, t in enumerate(tickers):
            ax.annotate(t, (asset_vols[i], mu[i]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    if tangency_weights is not None:
        # tangency point + capital market line (straight line from rf through
        # the tangency portfolio)
        t_ret = tangency_weights @ mu
        t_vol = float(np.sqrt(tangency_weights @ Sigma @ tangency_weights))
        ax.scatter([t_vol], [t_ret], color="tab:red", zorder=4, marker="*", s=150, label="Tangency portfolio")
        if t_vol > 0:
            cml_x = np.linspace(0, max(risks), 50)
            cml_y = rf + (t_ret - rf) / t_vol * cml_x
            ax.plot(cml_x, cml_y, linestyle="--", color="tab:red", alpha=0.6, label="Capital market line")

    ax.set_xlabel("Risk (annualized volatility)")
    ax.set_ylabel("Expected return (annualized)")
    ax.set_title("Efficient Frontier")
    ax.legend()
    return ax


def plot_covariance_heatmap(Sigma: np.ndarray, tickers: list[str], ax: plt.Axes | None = None) -> plt.Axes:
    # just a visual sanity check on Sigma before handing it to the optimizer 

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(Sigma, cmap="coolwarm")
    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.set_yticklabels(tickers)
    ax.set_title("Covariance Matrix")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_backtest_cumulative_returns(
    naive: pd.Series,
    shrinkage: pd.Series,
    equal_weight: pd.Series,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    # the "does it survive out-of-sample" picture ???
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    for series, label, color in [
        (naive, "Naive Markowitz", "tab:red"),
        (shrinkage, "Shrinkage Markowitz", "tab:blue"),
        (equal_weight, "Equal-weight (1/N)", "tab:gray"),
    ]:
        cumulative = (1 + series).cumprod()
        ax.plot(cumulative.index, cumulative.values, label=label, color=color)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative growth of $1")
    ax.set_title("Out-of-Sample Backtest")
    ax.legend()
    return ax
