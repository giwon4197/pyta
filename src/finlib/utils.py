from __future__ import annotations

import pandas as pd


def ensure_dataframe(data) -> pd.DataFrame:
    """Convert input data to a pandas DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(data)


def normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Normalize price series to start at 1.0."""
    return prices / prices.iloc[0]
