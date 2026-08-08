# closed-form solver + tangency portfolio (see README "Project structure" / "Why this project")
#
# from the README: "Closed-form mean-variance optimizer: solves for optimal weights
# directly from the Lagrangian system, not via a numerical solver (like the scipy
# module)". this file is that -- the actual point of the whole project, everything
# else (data.py, stats.py, shrinkage.py) just feeds mu/Sigma into this.
#
# derivation (so I don't have to re-derive this every time I forget):
#
#   minimize   (1/2) w^T Sigma w
#   subject to w^T mu = R        (hit a target return R)
#              w^T 1  = 1        (fully invested, weights sum to 1)
#
#   Lagrangian: L = (1/2) w^T Sigma w - l1(w^T mu - R) - l2(w^T 1 - 1)
#   dL/dw = Sigma w - l1 mu - l2 1 = 0  =>  w = l1 Sigma^-1 mu + l2 Sigma^-1 1
#
#   so every efficient portfolio is just a linear combo of two vectors:
#   a = Sigma^-1 mu and b = Sigma^-1 1. (this is "two-fund separation" --
#   the thing that makes closed form possible in the first place.)
#
#   define A = mu.a, B = mu.b (= 1.a), C = 1.b, D = A*C - B^2. plugging the two
#   constraints back in and solving the 2x2 system for l1, l2 gives:
#     l1 = (C*R - B) / D
#     l2 = (A - B*R) / D
#   -> w(R) = l1*a + l2*b. that's efficient_portfolio() below.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class FrontierPoint:
    target_return: float
    risk: float
    weights: np.ndarray


def _sigma_solve_vectors(mu: np.ndarray, Sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # key design decision (from README): "np.linalg.solve over np.linalg.inv
    # because matrix inversion increases numerical error when the covariance
    # matrix is near-singular. Therefore solving the linear system directly is
    # more stable and accurate." -- so never call np.linalg.inv(Sigma) anywhere,
    # solve for a and b directly instead.
    ones = np.ones_like(mu)
    a = np.linalg.solve(Sigma, mu)     # Sigma^-1 mu
    b = np.linalg.solve(Sigma, ones)   # Sigma^-1 1
    return a, b


def _abcd(mu: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    # A, B, C, D from the derivation above -- just the scalar building blocks
    # for l1, l2. nothing fancy, just names to match the math.
    A = mu @ a
    B = mu @ b
    C = np.ones_like(mu) @ b
    D = A * C - B**2
    return A, B, C, D


def efficient_portfolio(mu: np.ndarray, Sigma: np.ndarray, target_return: float) -> np.ndarray:
    """Closed-form minimum-variance weights for a given target return."""
    # reminder: this is w(R) = l1*a + l2*b from the derivation. no solver call,
    # no iteration -- just plugging R into the closed-form expression.
    a, b = _sigma_solve_vectors(mu, Sigma)
    A, B, C, D = _abcd(mu, a, b)
    l1 = (C * target_return - B) / D
    l2 = (A - B * target_return) / D
    return l1 * a + l2 * b


def min_variance_portfolio(Sigma: np.ndarray) -> np.ndarray:
    """Global minimum-variance portfolio (no return target, just min risk)."""
    # reminder: this is the special case of the frontier with no return
    # constraint at all -- just w proportional to Sigma^-1 * 1, normalized to
    # sum to 1. it's the leftmost tip of the efficient frontier (lowest risk
    # point), used in tests to sanity-check the frontier is shaped right.
    ones = np.ones(Sigma.shape[0])
    b = np.linalg.solve(Sigma, ones)
    return b / (ones @ b)


def tangency_portfolio(mu: np.ndarray, Sigma: np.ndarray, rf: float = 0.0) -> np.ndarray:
    """Max-Sharpe (tangency) portfolio, closed form.

    w_tan proportional to Sigma^-1 (mu - rf*1), normalized to sum to 1.
    """
    # from README Features: "Tangency (max Sharpe) portfolio is cross-validated
    # against a scipy.optimize solution for validation" -- this is the closed-
    # form half of that pair, tangency_portfolio_scipy() below is the check.
    # this is the portfolio shrinkage.py actually uses for the backtest.
    ones = np.ones_like(mu)
    excess = mu - rf * ones
    z = np.linalg.solve(Sigma, excess)
    return z / (ones @ z)


def tangency_portfolio_scipy(mu: np.ndarray, Sigma: np.ndarray, rf: float = 0.0) -> np.ndarray:
    """Numerically maximize Sharpe ratio via SLSQP, as a correctness check
    against the closed-form `tangency_portfolio` above."""
    # from README Key design decisions: "Closed-form implementation validated
    # against scipy.optimize (checking answers) -- not used as the primary
    # solver, just as a check on the hand-derived math." so this function is
    # NOT called anywhere except tests -- do not swap it in as the real solver.
    n = len(mu)
    w0 = np.ones(n) / n

    def neg_sharpe(w: np.ndarray) -> float:
        ret = w @ mu - rf
        vol = np.sqrt(w @ Sigma @ w)
        return -ret / vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    result = minimize(neg_sharpe, w0, method="SLSQP", constraints=constraints)
    if not result.success:
        raise RuntimeError(f"scipy tangency optimization failed: {result.message}")
    return result.x


def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    return float(weights @ mu)


def portfolio_vol(weights: np.ndarray, Sigma: np.ndarray) -> float:
    return float(np.sqrt(weights @ Sigma @ weights))


def portfolio_sharpe(weights: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, rf: float = 0.0) -> float:
    # reminder: Sharpe = excess return / risk. rf defaults to 0 so this still
    # works if I forget to pass a risk-free rate, but backtests should always
    # pass the real rf (see shrinkage.py / notebook, using 2% annualized).
    return (portfolio_return(weights, mu) - rf) / portfolio_vol(weights, Sigma)


def efficient_frontier(
    mu: np.ndarray,
    Sigma: np.ndarray,
    n_points: int = 50,
    return_range: tuple[float, float] | None = None,
) -> list[FrontierPoint]:
    """Trace the efficient frontier across a range of target returns."""
    # from README Features: "Efficient frontier construction across a range of
    # target returns (stocks outlined later)". just calls efficient_portfolio()
    # in a loop across target returns -- there's no cleverer way to do this
    # since each point is an independent closed-form solve.
    if return_range is None:
        lo, hi = float(mu.min()), float(mu.max())
        # widen slightly so the frontier extends past the individual assets
        pad = 0.25 * (hi - lo) if hi > lo else 0.05 * abs(hi)
        return_range = (lo - pad, hi + pad)

    targets = np.linspace(return_range[0], return_range[1], n_points)
    points = []
    for R in targets:
        w = efficient_portfolio(mu, Sigma, R)
        points.append(FrontierPoint(target_return=R, risk=portfolio_vol(w, Sigma), weights=w))
    return points
