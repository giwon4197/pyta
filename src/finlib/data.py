from __future__ import annotations

import pandas as pd


def load_price_data(path: str, parse_dates: bool = True) -> pd.DataFrame:
    """Load price data from a CSV file into a DataFrame."""
    return pd.read_csv(path, parse_dates=parse_dates, index_col=0)


def validate_price_data(prices: pd.DataFrame) -> None:
    """Validate that the price DataFrame is usable for calculations."""
    if prices.empty:
        raise ValueError("Price data must not be empty.")
    if prices.index.isnull().any():
        raise ValueError("Price index contains null values.")
    if prices.columns.duplicated().any():
        raise ValueError("Price columns must be unique.")
    if not prices.applymap(lambda x: isinstance(x, (int, float))).all().all():
        raise TypeError("Price data must contain only numeric values.")
