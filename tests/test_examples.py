from examples.sample_data import make_sample_ohlcv


def test_make_sample_ohlcv_shape_and_columns():
    df = make_sample_ohlcv(rows=30, seed=7)

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 30
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert df["volume"].gt(0).all()
