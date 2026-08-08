# correctness checks against scipy, sanity invariants (see README "Project structure")
#
# thsi fiel is the Test suite taht validates the closed-form solution against numerical methods and checking basic invariants (weights sum to 1, target
# return is hit)". 
# there are two categories:
#   1. invariants that have to hold no matter what (weights sum to 1, etc.)
#   2. closed-form vs scipy.optimize agreement, the actual "is my Lagrangian
#      derivation right" check from optimizer.py

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from src import optimizer

RNG = np.random.default_rng(42)


def _random_problem(n_assets: int = 5):
    # reminder: fixed seed (42) so these tests are reproducible

    raw = RNG.normal(size=(n_assets * 20, n_assets))
    Sigma = np.cov(raw, rowvar=False) + np.eye(n_assets) * 1e-3
    mu = RNG.uniform(0.02, 0.15, size=n_assets)
    return mu, Sigma


def test_efficient_portfolio_weights_sum_to_one():
    mu, Sigma = _random_problem()
    w = optimizer.efficient_portfolio(mu, Sigma, target_return=0.08)
    assert np.isclose(w.sum(), 1.0, atol=1e-8)


def test_efficient_portfolio_hits_target_return():

    mu, Sigma = _random_problem()
    target = 0.09
    w = optimizer.efficient_portfolio(mu, Sigma, target_return=target)
    assert np.isclose(w @ mu, target, atol=1e-8)


def test_efficient_portfolio_matches_scipy_numerical_solution():
    # the big one: closed-form Lagrangian solution vs. a numerical QP solve of
    # the exact same if fail 
    mu, Sigma = _random_problem()
    target = 0.10
    w_closed = optimizer.efficient_portfolio(mu, Sigma, target_return=target)

    n = len(mu)
    w0 = np.ones(n) / n
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: w @ mu - target},
    ]
    result = minimize(
        lambda w: w @ Sigma @ w,
        w0,
        method="SLSQP",
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500},
    )
    assert result.success
    np.testing.assert_allclose(w_closed, result.x, atol=1e-5)


def test_min_variance_portfolio_has_lowest_risk_on_frontier():
    # reminder: the GMV portfolio should be at (or below, floating point) the lowest-risk point anywhere on the frontier -
    mu, Sigma = _random_problem()
    w_gmv = optimizer.min_variance_portfolio(Sigma)
    gmv_risk = optimizer.portfolio_vol(w_gmv, Sigma)

    frontier = optimizer.efficient_frontier(mu, Sigma, n_points=100)
    frontier_risks = [p.risk for p in frontier]

    assert gmv_risk <= min(frontier_risks) + 1e-6


def test_tangency_portfolio_matches_scipy_max_sharpe():

    mu, Sigma = _random_problem()
    rf = 0.02

    w_closed = optimizer.tangency_portfolio(mu, Sigma, rf=rf)
    w_scipy = optimizer.tangency_portfolio_scipy(mu, Sigma, rf=rf)

    sharpe_closed = optimizer.portfolio_sharpe(w_closed, mu, Sigma, rf=rf)
    sharpe_scipy = optimizer.portfolio_sharpe(w_scipy, mu, Sigma, rf=rf)

    assert np.isclose(sharpe_closed, sharpe_scipy, atol=1e-4)
    np.testing.assert_allclose(w_closed, w_scipy, atol=1e-3)


def test_tangency_weights_sum_to_one():
    mu, Sigma = _random_problem()
    w = optimizer.tangency_portfolio(mu, Sigma, rf=0.01)
    assert np.isclose(w.sum(), 1.0, atol=1e-8)


def test_efficient_frontier_risk_increases_away_from_gmv():
    # reminder: risk should only go up as target return moves away from the GMV return in either direction.
    mu, Sigma = _random_problem()
    w_gmv = optimizer.min_variance_portfolio(Sigma)
    gmv_return = optimizer.portfolio_return(w_gmv, mu)

    frontier = optimizer.efficient_frontier(
        mu, Sigma, n_points=60, return_range=(gmv_return, gmv_return + 0.15)
    )
    risks = [p.risk for p in frontier]
    assert all(r2 >= r1 - 1e-8 for r1, r2 in zip(risks, risks[1:]))


def test_singular_covariance_raises():
    # a singular Sigma (e.g. two identical/duplicated assets) can't
    # be solved by np.linalg.solve 
    #  I'm NOT catching this and falling back to pinv or anything like that
    Sigma = np.array([[0.04, 0.04], [0.04, 0.04]])
    mu = np.array([0.08, 0.08])
    with pytest.raises(np.linalg.LinAlgError):
        optimizer.efficient_portfolio(mu, Sigma, target_return=0.08)
