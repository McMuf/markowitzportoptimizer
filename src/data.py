# price data fetching + local cachin
# everythig dpends on this file!!!!

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


def _cache_key(tickers: list[str], start: str, end: str) -> str:
    # reminder: hash (tickers, start, end) so each distinct request gets its own
    raw = f"{sorted(tickers)}|{start}|{end}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def get_prices(
    tickers: list[str],
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch adjusted close prices for `tickers` between `start` and `end`.

    Returns a DataFrame indexed by date, one column per ticker. Results are
    cached to disk (as parquet) so repeated runs/backtests don't re-hit the
    network for the same request.
    """
    # rem: yfinance calls are slow and I re-run the same tickers/date range  a lot while iterating in the notebook
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"prices_{_cache_key(tickers, start, end)}.parquet"

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    # auto_adjust=True -> prices are already split/dividend-adjusted, which is
    # what "adjusted close" means and what mu/Sigma should be computed from.
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    # yfinance quirk: multi-ticker downloads come back with MultiIndex columns
    # (field, ticker), but a single-ticker download collapses to flat columns.
    # handle both so get_prices() always returns the same shape regardless of how many tickers were passed in.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all").sort_index()

    if prices.empty:
        
 
        raise ValueError(f"No price data returned for {tickers} in [{start}, {end}]")   # propagate into stats.py (mean/cov of nothing is not a useful error).

    if use_cache:
        prices.to_parquet(cache_path)

    return prices


def get_returns(
    tickers: list[str],
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Daily simple returns computed from cached/fetched prices."""
    # reminder: simple returns (pct_change), not log returns stats.py jus takes the mean/cov of whatever comes out of here and annualizes it, so
    # keep it consistent with that (log returns would need a differentannualization / wouldn't sum the way portfolio weights expect).
    prices = get_prices(tickers, start, end, use_cache=use_cache)
    returns = prices.pct_change().dropna(how="all")
    return returns
