from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily simple returns from price data."""
    return prices.pct_change().dropna(how="all")


def cumulative_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate cumulative returns from returns series."""
    return (1 + returns).cumprod() - 1


def annualized_volatility(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    """Annualized volatility of return series."""
    return returns.std(ddof=1) * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.DataFrame, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> pd.Series:
    """Calculate the annualized Sharpe ratio for each series."""
    excess_returns = returns - risk_free_rate / periods_per_year
    ann_return = returns.mean() * periods_per_year
    ann_vol = annualized_volatility(returns, periods_per_year)
    return ann_return / ann_vol.replace(0, np.nan)


def max_drawdown(returns: pd.DataFrame) -> pd.Series:
    """Maximum drawdown from cumulative return series."""
    wealth_index = (1 + returns).cumprod()
    running_max = wealth_index.cummax()
    drawdown = wealth_index / running_max - 1
    return drawdown.min()


def moving_average(prices: pd.Series, window: int = 20) -> pd.Series:
    """Compute a rolling moving average."""
    return prices.rolling(window=window, min_periods=1).mean()
