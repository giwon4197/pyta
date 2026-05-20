import numpy as np
import pandas as pd
import pytest
import pyta

from pyta import (
    adx,
    atr,
    bollinger_bands,
    ema,
    institution_pressure_normalized_weight,
    macd,
    rsi,
    sma,
    spider_chart_indicator,
    stoch_rsi,
    stochastic_oscillator,
    supertrend,
)
from finlib import derived_indicators


def test_sma_and_ema_keep_index():
    prices = pd.Series([10.0, 11.0, 12.0, 13.0], index=pd.date_range("2026-01-01", periods=4))

    simple = sma(prices, length=3)
    exponential = ema(prices, length=3)

    assert simple.index.equals(prices.index)
    assert simple.iloc[-1] == pytest.approx(12.0)
    assert exponential.iloc[-1] > simple.iloc[-2]


def test_rsi_is_bounded_after_warmup():
    prices = pd.Series([44, 45, 46, 45, 47, 48, 47, 49, 50, 51, 50, 52, 53, 54, 55], dtype=float)

    value = rsi(prices, length=5).dropna()

    assert not value.empty
    assert value.between(0, 100).all()


def test_macd_returns_expected_columns():
    prices = pd.Series(range(1, 60), dtype=float)

    result = macd(prices, 12, 26, 9)

    assert list(result.columns) == ["macd", "signal", "histogram"]
    assert result["histogram"].dropna().iloc[-1] == pytest.approx(
        result["macd"].dropna().iloc[-1] - result["signal"].dropna().iloc[-1]
    )


def test_bollinger_bands_wrap_price_series():
    prices = pd.Series([10.0, 12.0, 14.0, 16.0])

    bands = bollinger_bands(prices, 2, 2)

    assert list(bands.columns) == ["middle", "upper", "lower"]
    assert bands["upper"].iloc[-1] > bands["middle"].iloc[-1] > bands["lower"].iloc[-1]


def test_atr_and_stochastic_oscillator():
    high = pd.Series([10, 11, 12, 13, 14], dtype=float)
    low = pd.Series([8, 9, 10, 11, 12], dtype=float)
    close = pd.Series([9, 10, 11, 12, 13], dtype=float)

    average_range = atr(high, low, close, length=3)
    stochastic = stochastic_oscillator(high, low, close, k_length=3, d_length=2)

    assert average_range.dropna().iloc[-1] > 0
    assert list(stochastic.columns) == ["k", "d"]
    assert stochastic["k"].dropna().between(0, 100).all()


def test_adx_returns_directional_columns():
    high = pd.Series(
        [30, 32, 33, 35, 34, 36, 38, 37, 39, 41, 42, 43, 45, 44, 46, 48],
        dtype=float,
    )
    low = pd.Series(
        [28, 29, 31, 32, 31, 33, 35, 34, 36, 38, 39, 40, 42, 41, 43, 45],
        dtype=float,
    )
    close = pd.Series(
        [29, 31, 32, 34, 33, 35, 37, 36, 38, 40, 41, 42, 44, 43, 45, 47],
        dtype=float,
    )

    result = adx(high, low, close, length=5)

    assert list(result.columns) == ["plus_di", "minus_di", "adx"]
    assert result["adx"].dropna().between(0, 100).all()
    assert result["plus_di"].dropna().iloc[-1] > result["minus_di"].dropna().iloc[-1]


def test_supertrend_accepts_pine_style_positional_inputs():
    high = pd.Series(
        [30, 32, 33, 35, 34, 36, 38, 37, 39, 41, 42, 43, 45, 44, 46, 48],
        dtype=float,
    )
    low = pd.Series(
        [28, 29, 31, 32, 31, 33, 35, 34, 36, 38, 39, 40, 42, 41, 43, 45],
        dtype=float,
    )
    close = pd.Series(
        [29, 31, 32, 34, 33, 35, 37, 36, 38, 40, 41, 42, 44, 43, 45, 47],
        dtype=float,
    )

    result = supertrend(high, low, close, 3, 5)

    assert list(result.columns) == ["supertrend", "direction", "upper_band", "lower_band"]
    assert result["direction"].dropna().isin([-1.0, 1.0]).all()
    assert not result["supertrend"].dropna().empty


def test_institution_pressure_normalized_weight_matches_expected_shape():
    index = pd.RangeIndex(180)
    open_ = pd.Series(np.linspace(100, 130, 180), index=index)
    high = open_ + 3
    low = open_ - 3
    close = open_ + pd.Series(np.sin(np.arange(180) / 6), index=index)
    volume = pd.Series(1000 + (np.arange(180) % 11) * 100, index=index, dtype=float)

    result = institution_pressure_normalized_weight(
        open_,
        high,
        low,
        close,
        volume,
        vol_len=20,
        range_len=30,
        sum_len=10,
        bb_len=20,
        bb_mult=2,
    )

    assert list(result.columns) == [
        "normalized_weight",
        "institution_pressure",
        "accum_score",
        "dist_score",
        "basis",
        "upper",
        "lower",
        "vol_z",
        "move_z",
        "price_pos",
    ]
    assert result.index.equals(index)
    assert result["normalized_weight"].between(0, 100).all()
    assert (
        institution_pressure_normalized_weight
        is derived_indicators.institution_pressure_normalized_weight
    )


def test_stoch_rsi_matches_tradingview_shape():
    close = pd.Series(
        [100, 101, 102, 101, 103, 104, 105, 103, 106, 108, 109, 110,
         108, 111, 113, 114, 115, 116, 114, 117, 118, 119, 120, 121,
         119, 122, 123, 124, 125, 126, 127, 128, 126, 129, 130],
        dtype=float,
    )

    result = stoch_rsi(close, 3, 3, 14, 14)

    assert list(result.columns) == ["k", "d", "rsi", "stoch_rsi"]
    assert result["k"].dropna().between(0, 100).all()
    assert result["d"].dropna().between(0, 100).all()
    assert stoch_rsi is derived_indicators.stoch_rsi


def test_spider_chart_indicator_returns_scores_and_signals():
    index = pd.RangeIndex(80)
    close = pd.Series(
        np.linspace(100, 130, 80) + np.sin(np.arange(80) / 3),
        index=index,
    )
    high = close + 2
    low = close - 2
    volume = pd.Series(1000 + (np.arange(80) % 9) * 50, index=index, dtype=float)

    result = spider_chart_indicator(
        high,
        low,
        close,
        volume,
        rsi_len=10,
        stoch_len=10,
        cor_len=10,
        mfi_len=10,
        wpr_len=10,
        pr_len=10,
        cmo_len=10,
        aos_len=10,
    )

    assert list(result.columns) == [
        "avg_score",
        "rsi",
        "stoch",
        "cor",
        "mfi",
        "wpr",
        "pr",
        "cmo",
        "aos",
        "long_entry",
        "short_entry",
        "long_stop",
        "short_stop",
    ]
    assert result["avg_score"].dropna().between(0, 1).all()
    assert result["long_entry"].dtype == bool
    assert spider_chart_indicator is derived_indicators.spider_chart_indicator


def test_popular_standard_indicators_are_available():
    index = pd.RangeIndex(90)
    close = pd.Series(
        np.linspace(100, 150, 90) + np.sin(np.arange(90) / 4),
        index=index,
    )
    high = close + 2
    low = close - 2
    volume = pd.Series(1000 + (np.arange(90) % 13) * 100, index=index, dtype=float)

    assert pyta.obv(close, volume).index.equals(index)
    assert pyta.vwap(high, low, close, volume).dropna().iloc[-1] > 0
    assert pyta.mfi(high, low, close, volume, length=14).dropna().between(0, 100).all()
    assert not pyta.cci(high, low, close, length=20).dropna().empty
    assert not pyta.roc(close, length=10).dropna().empty
    assert not pyta.mom(close, length=10).dropna().empty
    assert not pyta.trix(close, length=5).dropna().empty
    assert not pyta.tsi(close).dropna().empty
    assert pyta.uo(high, low, close).dropna().between(0, 100).all()
    assert list(pyta.aroon(high, low, length=14).columns) == ["aroon_up", "aroon_down"]
    assert not pyta.sar(high, low, close).dropna().empty

    ichimoku = pyta.ichimoku(high, low, close)
    assert list(ichimoku.columns) == [
        "tenkan_sen",
        "kijun_sen",
        "senkou_span_a",
        "senkou_span_b",
        "chikou_span",
    ]

    assert not pyta.stddev(close, length=10).dropna().empty
    assert not pyta.variance(close, length=10).dropna().empty
    assert not pyta.historical_volatility(close, length=10).dropna().empty
    assert pyta.natr(high, low, close).dropna().ge(0).all()
    assert pyta.chop(high, low, close).dropna().between(0, 100).all()

    assert not pyta.ad(high, low, close, volume).dropna().empty
    assert not pyta.cmf(high, low, close, volume).dropna().empty
    assert not pyta.adosc(high, low, close, volume).dropna().empty
    assert not pyta.eom(high, low, volume).dropna().empty
    assert not pyta.pvt(close, volume).dropna().empty

    assert list(pyta.pivot_points(high, low, close).columns) == [
        "pivot",
        "r1",
        "s1",
        "r2",
        "s2",
        "r3",
        "s3",
    ]
    assert pyta.donchian_channel is pyta.dc
    assert pyta.keltner_channel is pyta.kc
    assert pyta.williams_r is pyta.wpr
    assert pyta.psar is pyta.sar
    assert pyta.ultimate_oscillator is pyta.uo
    assert pyta.stdev is pyta.stddev


def test_dataframe_return_column_rules():
    index = pd.RangeIndex(90)
    close = pd.Series(np.linspace(100, 150, 90), index=index)
    high = close + 2
    low = close - 2
    volume = pd.Series(1000 + (np.arange(90) % 7) * 100, index=index, dtype=float)

    expected_columns = {
        "macd": ["macd", "signal", "histogram"],
        "stoch": ["k", "d"],
        "adx": ["plus_di", "minus_di", "adx"],
        "bb": ["middle", "upper", "lower"],
        "kc": ["middle", "upper", "lower"],
        "dc": ["middle", "upper", "lower"],
        "supertrend": ["supertrend", "direction", "upper_band", "lower_band"],
        "aroon": ["aroon_up", "aroon_down"],
        "ichimoku": [
            "tenkan_sen",
            "kijun_sen",
            "senkou_span_a",
            "senkou_span_b",
            "chikou_span",
        ],
        "pivot_points": ["pivot", "r1", "s1", "r2", "s2", "r3", "s3"],
        "stoch_rsi": ["k", "d", "rsi", "stoch_rsi"],
        "spider_chart": [
            "avg_score",
            "rsi",
            "stoch",
            "cor",
            "mfi",
            "wpr",
            "pr",
            "cmo",
            "aos",
            "long_entry",
            "short_entry",
            "long_stop",
            "short_stop",
        ],
        "whale_tracking": [
            "normalized_weight",
            "institution_pressure",
            "accum_score",
            "dist_score",
            "basis",
            "upper",
            "lower",
            "vol_z",
            "move_z",
            "price_pos",
        ],
    }

    actual = {
        "macd": pyta.macd(close).columns.tolist(),
        "stoch": pyta.stoch(high, low, close).columns.tolist(),
        "adx": pyta.adx(high, low, close).columns.tolist(),
        "bb": pyta.bb(close).columns.tolist(),
        "kc": pyta.kc(high, low, close).columns.tolist(),
        "dc": pyta.dc(high, low, close).columns.tolist(),
        "supertrend": pyta.supertrend(high, low, close).columns.tolist(),
        "aroon": pyta.aroon(high, low).columns.tolist(),
        "ichimoku": pyta.ichimoku(high, low, close).columns.tolist(),
        "pivot_points": pyta.pivot_points(high, low, close).columns.tolist(),
        "stoch_rsi": pyta.stoch_rsi(close).columns.tolist(),
        "spider_chart": pyta.spider_chart(high, low, close, volume).columns.tolist(),
        "whale_tracking": pyta.whale_tracking(close, high, low, close, volume).columns.tolist(),
    }

    assert actual == expected_columns


def test_window_aliases_still_work():
    prices = pd.Series([10.0, 12.0, 14.0, 16.0])
    high = pd.Series([10, 11, 12, 13, 14], dtype=float)
    low = pd.Series([8, 9, 10, 11, 12], dtype=float)
    close = pd.Series([9, 10, 11, 12, 13], dtype=float)

    pd.testing.assert_series_equal(sma(prices, length=2), sma(prices, window=2))
    pd.testing.assert_series_equal(rsi(prices, length=2), rsi(prices, window=2))
    pd.testing.assert_frame_equal(
        bollinger_bands(prices, length=2),
        bollinger_bands(prices, window=2),
    )
    pd.testing.assert_series_equal(
        atr(high, low, close, length=3),
        atr(high, low, close, window=3),
    )
    pd.testing.assert_frame_equal(
        macd(pd.Series(range(1, 60), dtype=float), fast_length=12),
        macd(pd.Series(range(1, 60), dtype=float), fast=12),
    )
    pd.testing.assert_frame_equal(
        bollinger_bands(prices, length=2, mult=2),
        bollinger_bands(prices, length=2, num_std=2),
    )
