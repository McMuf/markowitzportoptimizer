"""Price data fetching with local disk caching."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


def _cache_key(tickers: list[str], start: str, end: str) -> str:
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
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"prices_{_cache_key(tickers, start, end)}.parquet"

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # single-ticker download collapses to a flat frame
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all").sort_index()

    if prices.empty:
        raise ValueError(f"No price data returned for {tickers} in [{start}, {end}]")

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
    prices = get_prices(tickers, start, end, use_cache=use_cache)
    returns = prices.pct_change().dropna(how="all")
    return returns
