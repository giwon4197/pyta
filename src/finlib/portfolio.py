from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_return(returns: pd.DataFrame, weights: list[float]) -> pd.Series:
    """Calculate portfolio returns given weights and asset returns."""
    if len(weights) != returns.shape[1]:
        raise ValueError("Number of weights must match number of assets.")
    weights_arr = np.asarray(weights, dtype=float)
    return returns.dot(weights_arr)


def portfolio_volatility(returns: pd.DataFrame, weights: list[float], periods_per_year: int = 252) -> float:
    """Calculate annualized portfolio volatility."""
    if len(weights) != returns.shape[1]:
        raise ValueError("Number of weights must match number of assets.")
    cov = returns.cov() * periods_per_year
    weights_arr = np.asarray(weights, dtype=float)
    return float(np.sqrt(weights_arr.T @ cov.values @ weights_arr))
