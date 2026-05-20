from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def whale_tracking(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    vol_len: int = 100,
    range_len: int = 120,
    sum_len: int = 24,
    bb_len: int = 100,
    bb_mult: float = 2.0,
) -> pd.DataFrame:
    """Institution Pressure Normalized Weight V2 translated from Pine Script."""
    ind._validate_length(vol_len, "vol_len")
    ind._validate_length(range_len, "range_len")
    ind._validate_length(sum_len, "sum_len")
    ind._validate_length(bb_len, "bb_len")

    trade_value = close * volume

    avg_value = ind.sma(trade_value, length=vol_len)
    dev_value = trade_value.rolling(window=vol_len, min_periods=vol_len).std(ddof=0)
    vol_z = pd.Series(
        np.where(dev_value > 0, (trade_value - avg_value) / dev_value, 0.0),
        index=close.index,
    )

    body_move = (close - open_).abs() / open_.replace(0, np.nan)
    avg_move = ind.sma(body_move, length=vol_len)
    dev_move = body_move.rolling(window=vol_len, min_periods=vol_len).std(ddof=0)
    move_z = pd.Series(
        np.where(dev_move > 0, (body_move - avg_move) / dev_move, 0.0),
        index=close.index,
    )

    range_high = high.rolling(window=range_len, min_periods=range_len).max()
    range_low = low.rolling(window=range_len, min_periods=range_len).min()
    range_size = range_high - range_low
    price_pos = pd.Series(
        np.where(range_size > 0, (close - range_low) / range_size, 0.5),
        index=close.index,
    )

    is_low_zone = price_pos < 0.35
    is_mid_zone = (price_pos >= 0.35) & (price_pos <= 0.65)
    is_high_zone = price_pos > 0.65

    weight = pd.Series(
        np.select(
            [vol_z < 0.0, vol_z < 1.0, vol_z < 2.0, vol_z < 3.0],
            [1.0, 2.0, 3.0, 4.0],
            default=5.0,
        ),
        index=close.index,
    )

    is_accum = (is_low_zone | is_mid_zone) & (vol_z >= 0) & (move_z < 0.5)
    is_dist = is_high_zone & (vol_z >= 0) & (move_z < 0.5)

    accum_raw = pd.Series(np.where(is_accum, trade_value * weight, 0.0), index=close.index)
    dist_raw = pd.Series(np.where(is_dist, trade_value * weight, 0.0), index=close.index)

    accum_sum = accum_raw.rolling(window=sum_len, min_periods=sum_len).sum()
    dist_sum = dist_raw.rolling(window=sum_len, min_periods=sum_len).sum()

    scale = avg_value * sum_len
    accum_score = pd.Series(
        np.where(scale > 0, accum_sum / scale * 100, 0.0),
        index=close.index,
    )
    dist_score = pd.Series(
        np.where(scale > 0, dist_sum / scale * 100, 0.0),
        index=close.index,
    )

    institution_pressure = accum_score - dist_score

    basis = ind.sma(institution_pressure, length=bb_len)
    dev = institution_pressure.rolling(window=bb_len, min_periods=bb_len).std(ddof=0)
    upper = basis + dev * bb_mult
    lower = basis - dev * bb_mult
    band_range = upper - lower

    pos_ratio_raw = pd.Series(
        np.where(band_range > 0, (institution_pressure - lower) / band_range, 0.0),
        index=close.index,
    )
    normalized_weight = pos_ratio_raw.clip(lower=0, upper=1) * 100

    return pd.DataFrame(
        {
            "normalized_weight": normalized_weight,
            "institution_pressure": institution_pressure,
            "accum_score": accum_score,
            "dist_score": dist_score,
            "basis": basis,
            "upper": upper,
            "lower": lower,
            "vol_z": vol_z,
            "move_z": move_z,
            "price_pos": price_pos,
        }
    )


institution_pressure_normalized_weight = whale_tracking


def stoch_rsi(
    src: pd.Series,
    smooth_k: int = 3,
    smooth_d: int = 3,
    length_rsi: int = 14,
    length_stoch: int = 14,
) -> pd.DataFrame:
    """Stochastic RSI translated from TradingView's built-in Pine Script."""
    ind._validate_length(smooth_k, "smooth_k")
    ind._validate_length(smooth_d, "smooth_d")
    ind._validate_length(length_rsi, "length_rsi")
    ind._validate_length(length_stoch, "length_stoch")

    rsi_values = ind.rsi(src, length=length_rsi)
    lowest_rsi = rsi_values.rolling(
        window=length_stoch, min_periods=length_stoch
    ).min()
    highest_rsi = rsi_values.rolling(
        window=length_stoch, min_periods=length_stoch
    ).max()
    rsi_range = (highest_rsi - lowest_rsi).replace(0, np.nan)

    raw_k = 100 * (rsi_values - lowest_rsi) / rsi_range
    k = ind.sma(raw_k, length=smooth_k)
    d = ind.sma(k, length=smooth_d)

    return pd.DataFrame(
        {
            "k": k,
            "d": d,
            "rsi": rsi_values,
            "stoch_rsi": raw_k,
        }
    )


def _bars_ago_highest(values: np.ndarray) -> float:
    if np.isnan(values).any():
        return np.nan
    return float(len(values) - 1 - np.argmax(values))


def _bars_ago_lowest(values: np.ndarray) -> float:
    if np.isnan(values).any():
        return np.nan
    return float(len(values) - 1 - np.argmin(values))


def spider_chart(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    rsi_len: int = 14,
    stoch_len: int = 14,
    cor_len: int = 14,
    mfi_len: int = 14,
    wpr_len: int = 14,
    pr_len: int = 14,
    cmo_len: int = 14,
    aos_len: int = 14,
    *,
    rsi_tog: bool = True,
    stoch_tog: bool = True,
    cor_tog: bool = True,
    mfi_tog: bool = True,
    wpr_tog: bool = True,
    pr_tog: bool = True,
    cmo_tog: bool = True,
    aos_tog: bool = True,
    glen_tog: bool = False,
    glen: int = 14,
    long_line: float = 0.60,
    short_line: float = 0.40,
    long_stop_line: float = 0.60,
    short_stop_line: float = 0.40,
) -> pd.DataFrame:
    """Spider Chart oscillator score translated from Pine Script without chart drawing."""
    lengths = {
        "rsi_len": rsi_len,
        "stoch_len": stoch_len,
        "cor_len": cor_len,
        "mfi_len": mfi_len,
        "wpr_len": wpr_len,
        "pr_len": pr_len,
        "cmo_len": cmo_len,
        "aos_len": aos_len,
        "glen": glen,
    }
    for name, value in lengths.items():
        ind._validate_length(value, name)

    rsi_length = glen if glen_tog else rsi_len
    stoch_length = glen if glen_tog else stoch_len
    cor_length = glen if glen_tog else cor_len
    mfi_length = glen if glen_tog else mfi_len
    wpr_length = glen if glen_tog else wpr_len
    pr_length = glen if glen_tog else pr_len
    cmo_length = glen if glen_tog else cmo_len
    aos_length = glen if glen_tog else aos_len

    index = close.index
    src = close
    bar_index = pd.Series(np.arange(len(close), dtype=float), index=index)

    rsi_score = ind.rsi(src, length=rsi_length) / 100

    stoch_raw = ind.stoch(high, low, src, k_length=stoch_length, d_length=1)["k"]
    stoch_score = stoch_raw / 100

    corr = src.rolling(window=cor_length, min_periods=cor_length).corr(bar_index)
    cor_score = 0.5 * corr + 0.5

    hlc3 = (high + low + close) / 3
    positive_money = volume * hlc3.where(hlc3 > hlc3.shift(1), 0.0)
    total_money = volume * hlc3
    mfi_score = ind.sma(positive_money, length=mfi_length) / ind.sma(
        total_money, length=mfi_length
    ).replace(0, np.nan)

    wpr_score = ind.wpr(high, low, close, length=wpr_length) / 100 + 1

    change = src.diff()
    pr_score = change.pipe(np.sign).clip(lower=0).rolling(
        window=pr_length, min_periods=pr_length
    ).mean()

    positive_change_sum = change.clip(lower=0).rolling(
        window=cmo_length, min_periods=cmo_length
    ).sum()
    absolute_change_sum = change.abs().rolling(
        window=cmo_length, min_periods=cmo_length
    ).sum()
    cmo_score = positive_change_sum / absolute_change_sum.replace(0, np.nan)

    aos_window = aos_length + 1
    highest_bars = high.rolling(window=aos_window, min_periods=aos_window).apply(
        _bars_ago_highest, raw=True
    )
    lowest_bars = low.rolling(window=aos_window, min_periods=aos_window).apply(
        _bars_ago_lowest, raw=True
    )
    aos_score = (highest_bars - lowest_bars) / (aos_length * 2) + 0.5

    enabled_count = sum(
        [
            rsi_tog,
            stoch_tog,
            cor_tog,
            mfi_tog,
            wpr_tog,
            pr_tog,
            cmo_tog,
            aos_tog,
        ]
    )
    if enabled_count == 0:
        raise ValueError("At least one oscillator toggle must be enabled.")

    rsi_sig = rsi_score if rsi_tog else pd.Series(0.0, index=index)
    stoch_sig = stoch_score if stoch_tog else pd.Series(0.0, index=index)
    cor_sig = cor_score if cor_tog else pd.Series(0.0, index=index)
    mfi_sig = mfi_score if mfi_tog else pd.Series(0.0, index=index)
    wpr_sig = wpr_score if wpr_tog else pd.Series(0.0, index=index)
    pr_sig = pr_score if pr_tog else pd.Series(0.0, index=index)
    cmo_sig = cmo_score if cmo_tog else pd.Series(0.0, index=index)
    aos_sig = aos_score if aos_tog else pd.Series(0.0, index=index)

    avg_score = (
        rsi_sig
        + stoch_sig
        + cor_sig
        + mfi_sig
        + wpr_sig
        + pr_sig
        + cmo_sig
        + aos_sig
    ) / enabled_count

    long_entry = avg_score >= long_line
    short_entry = avg_score <= short_line
    long_stop = (avg_score.shift(1) >= long_stop_line) & (avg_score < long_stop_line)
    short_stop = (avg_score.shift(1) <= short_stop_line) & (avg_score > short_stop_line)

    return pd.DataFrame(
        {
            "avg_score": avg_score,
            "rsi": rsi_sig,
            "stoch": stoch_sig,
            "cor": cor_sig,
            "mfi": mfi_sig,
            "wpr": wpr_sig,
            "pr": pr_sig,
            "cmo": cmo_sig,
            "aos": aos_sig,
            "long_entry": long_entry,
            "short_entry": short_entry,
            "long_stop": long_stop,
            "short_stop": short_stop,
        }
    )


spider_chart_indicator = spider_chart
