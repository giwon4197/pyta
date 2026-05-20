from __future__ import annotations

import numpy as np
import pandas as pd


def make_sample_ohlcv(rows: int = 240, seed: int = 42) -> pd.DataFrame:
    """Create deterministic OHLCV sample data for examples and manual tests."""
    if rows <= 0:
        raise ValueError("rows must be greater than 0.")

    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=rows, freq="D")

    trend = np.linspace(100, 145, rows)
    cycle = 4 * np.sin(np.arange(rows) / 8)
    noise = rng.normal(0, 1.2, rows).cumsum() * 0.15
    close = pd.Series(trend + cycle + noise, index=index, name="close")

    open_ = close.shift(1).fillna(close.iloc[0]) + rng.normal(0, 0.8, rows)
    high = pd.concat([open_, close], axis=1).max(axis=1) + rng.uniform(0.4, 2.4, rows)
    low = pd.concat([open_, close], axis=1).min(axis=1) - rng.uniform(0.4, 2.4, rows)
    volume = pd.Series(
        rng.integers(800_000, 2_500_000, rows),
        index=index,
        name="volume",
        dtype=float,
    )

    return pd.DataFrame(
        {
            "open": open_.astype(float),
            "high": high.astype(float),
            "low": low.astype(float),
            "close": close.astype(float),
            "volume": volume,
        },
        index=index,
    )


if __name__ == "__main__":
    print(make_sample_ohlcv().head())
