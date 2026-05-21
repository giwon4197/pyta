from __future__ import annotations

import numpy as np
import pandas as pd

# Validation helpers

def _validate_length(length: int, name: str = "length") -> None:
    if length <= 0:
        raise ValueError(f"{name} must be greater than 0.")

def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")

# Rolling-window helpers

def _bars_ago_highest(values: np.ndarray) -> float:
    if np.isnan(values).any():
        return np.nan
    return float(len(values) - 1 - np.argmax(values))

def _bars_ago_lowest(values: np.ndarray) -> float:
    if np.isnan(values).any():
        return np.nan
    return float(len(values) - 1 - np.argmin(values))

def _hurst_exponent_window(
    values: np.ndarray,
    min_lag: int,
    max_lag: int,
) -> float:
    if np.isnan(values).any():
        return np.nan

    effective_max_lag = min(max_lag, len(values) - 1)
    lags = np.arange(min_lag, effective_max_lag + 1)
    if len(lags) < 2:
        return np.nan

    tau = np.array([np.std(values[lag:] - values[:-lag], ddof=0) for lag in lags])
    valid = tau > 0
    if valid.sum() < 2:
        return np.nan

    slope, _ = np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)
    return float(slope)

def _rma(series: pd.Series, length: int) -> pd.Series:
    rolling_mean = series.rolling(window=length, min_periods=length).mean()
    result = pd.Series(np.nan, index=series.index, dtype=float)

    for i, value in enumerate(series):
        if pd.isna(value):
            continue

        if i == 0 or pd.isna(result.iloc[i - 1]):
            if pd.notna(rolling_mean.iloc[i]):
                result.iloc[i] = rolling_mean.iloc[i]
            continue

        result.iloc[i] = (result.iloc[i - 1] * (length - 1) + value) / length

    return result

# Price and candle basics

def log_return(prices: pd.Series) -> pd.Series:
    """Log return between consecutive prices."""
    return np.log(prices / prices.shift(1))

def high_low_range(high: pd.Series, low: pd.Series) -> pd.Series:
    """High-low price range for each period."""
    return high - low

def body_ratio(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Candlestick body size as a ratio of the high-low range."""
    price_range = high_low_range(high, low).replace(0, np.nan)
    return (close - open_).abs() / price_range

# Moving averages

def sma(
    series: pd.Series,
    length: int = 20,
    min_periods: int | None = None,
    *,
    window: int | None = None,
) -> pd.Series:
    """Simple moving average."""
    length = window if window is not None else length
    _validate_length(length)
    return series.rolling(window=length, min_periods=min_periods or length).mean()

def ema(
    series: pd.Series,
    length: int = 20,
    min_periods: int | None = None,
    *,
    span: int | None = None,
) -> pd.Series:
    """Exponential moving average."""
    length = span if span is not None else length
    _validate_length(length)
    return series.ewm(span=length, adjust=False, min_periods=min_periods or length).mean()

def wma(
    series: pd.Series,
    length: int = 20,
    min_periods: int | None = None,
    *,
    window: int | None = None,
) -> pd.Series:
    """Weighted moving average."""
    length = window if window is not None else length
    _validate_length(length)
    weights = np.arange(1, length + 1)
    return series.rolling(window=length, min_periods=min_periods or length).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )

def vwma(
    price: pd.Series,
    volume: pd.Series,
    length: int = 20,
    min_periods: int | None = None,
    *,
    window: int | None = None,
) -> pd.Series:
    """Volume-weighted moving average."""
    length = window if window is not None else length
    _validate_length(length)
    weighted_price = price * volume
    sum_weighted_price = weighted_price.rolling(window=length, min_periods=min_periods or length).sum()
    sum_volume = volume.rolling(window=length, min_periods=min_periods or length).sum()
    return sum_weighted_price / sum_volume.replace(0, np.nan)

def hma(
    series: pd.Series,
    length: int = 20,
    min_periods: int | None = None,
    *,
    window: int | None = None,
) -> pd.Series:
    """Hull moving average."""
    length = window if window is not None else length
    _validate_length(length)
    half_length = max(1, length // 2)
    sqrt_length = max(1, int(np.sqrt(length)))
    wma_half = wma(series, length=half_length, min_periods=min_periods)
    wma_full = wma(series, length=length, min_periods=min_periods)
    diff = 2 * wma_half - wma_full
    return wma(diff, length=sqrt_length, min_periods=min_periods)  

def swma(
    series: pd.Series,
    length: int = 20,
    min_periods: int | None = None,
    *,
    window: int | None = None,
) -> pd.Series:
    """Smoothed moving average."""
    length = window if window is not None else length
    _validate_length(length)
    return series.ewm(alpha=1 / length, adjust=False, min_periods=min_periods or length).mean() 

# Statistics and volatility

def variance(
    series: pd.Series,
    length: int = 20,
    *,
    window: int | None = None,
    ddof: int = 0,
) -> pd.Series:
    """Rolling variance."""
    length = window if window is not None else length
    _validate_length(length)
    return series.rolling(window=length, min_periods=length).var(ddof=ddof)

def stddev(
    series: pd.Series,
    length: int = 20,
    *,
    window: int | None = None,
    ddof: int = 0,
) -> pd.Series:
    """Rolling standard deviation."""
    length = window if window is not None else length
    _validate_length(length)
    return series.rolling(window=length, min_periods=length).std(ddof=ddof)

stdev = stddev

def historical_volatility(
    prices: pd.Series,
    length: int = 20,
    periods_per_year: int = 252,
    *,
    window: int | None = None,
) -> pd.Series:
    """Annualized historical volatility from log returns."""
    length = window if window is not None else length
    _validate_length(length)
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.rolling(window=length, min_periods=length).std(ddof=1) * np.sqrt(
        periods_per_year
    )

def hurst_exponent(
    series: pd.Series,
    length: int = 100,
    min_lag: int = 2,
    max_lag: int = 20,
    *,
    window: int | None = None,
) -> pd.Series:
    """Rolling Hurst exponent estimated from log-log lagged differences."""
    length = window if window is not None else length
    _validate_length(length)
    _validate_length(min_lag, "min_lag")
    _validate_length(max_lag, "max_lag")
    if min_lag >= max_lag:
        raise ValueError("min_lag must be less than max_lag.")

    return series.rolling(window=length, min_periods=length).apply(
        lambda values: _hurst_exponent_window(values, min_lag, max_lag),
        raw=True,
    )

HistoricalVolatility = historical_volatility

HurstExponent = hurst_exponent

# Range and trend strength

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range for OHLC price data."""
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)

def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    *,
    window: int | None = None,
) -> pd.Series:
    """Average True Range using Wilder-style smoothing."""
    length = window if window is not None else length
    _validate_length(length)
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

def natr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    *,
    window: int | None = None,
) -> pd.Series:
    """Normalized Average True Range."""
    length = window if window is not None else length
    _validate_length(length)
    return 100 * atr(high, low, close, length=length) / close.replace(0, np.nan)

def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    *,
    window: int | None = None,
) -> pd.DataFrame:
    """Average Directional Index with +DI and -DI lines."""
    length = window if window is not None else length
    _validate_length(length)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=low.index,
    )

    average_true_range = atr(high, low, close, length=length)
    smoothed_plus_dm = plus_dm.ewm(
        alpha=1 / length, adjust=False, min_periods=length
    ).mean()
    smoothed_minus_dm = minus_dm.ewm(
        alpha=1 / length, adjust=False, min_periods=length
    ).mean()

    plus_di = 100 * smoothed_plus_dm / average_true_range.replace(0, np.nan)
    minus_di = 100 * smoothed_minus_dm / average_true_range.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    return pd.DataFrame(
        {
            "plus_di": plus_di,
            "minus_di": minus_di,
            "adx": adx_line,
        }
    )

def aroon(
    high: pd.Series,
    low: pd.Series,
    length: int = 14,
    *,
    window: int | None = None,
) -> pd.DataFrame:
    """Aroon indicator with Aroon Up and Aroon Down lines."""
    length = window if window is not None else length
    _validate_length(length)
    highest_high = high.rolling(window=length, min_periods=length).apply(_bars_ago_highest, raw=True)
    lowest_low = low.rolling(window=length, min_periods=length).apply(_bars_ago_lowest, raw=True)

    aroon_up = 100 * (length - highest_high) / length
    aroon_down = 100 * (length - lowest_low) / length

    return pd.DataFrame(
        {
            "aroon_up": aroon_up,
            "aroon_down": aroon_down,
        }
    )

def sar(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    step: float = 0.02,
    max_step: float = 0.2,
    *,
    step_increment: float | None = None,
    max_step_limit: float | None = None,
) -> pd.Series:
    """Parabolic SAR indicator."""
    step = step_increment if step_increment is not None else step
    max_step = max_step_limit if max_step_limit is not None else max_step
    _validate_positive(step, "step")
    _validate_positive(max_step, "max_step")

    sar_values = pd.Series(np.nan, index=close.index, dtype=float)
    trend = 1  # 1 for uptrend, -1 for downtrend
    ep = high.iloc[0]
    af = step  # Acceleration Factor

    sar_values.iloc[0] = low.iloc[0] - af * (high.iloc[0] - low.iloc[0])

    for i in range(1, len(close)):
        prev_sar = sar_values.iloc[i - 1]
        prev_high = high.iloc[i - 1]
        prev_low = low.iloc[i - 1]

        if trend == 1:
            sar_values.iloc[i] = prev_sar + af * (ep - prev_sar)
            if high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + step, max_step)
            if low.iloc[i] < sar_values.iloc[i]:
                trend = -1
                sar_values.iloc[i] = ep
                ep = low.iloc[i]
                af = step
        else:
            sar_values.iloc[i] = prev_sar + af * (ep - prev_sar)
            if low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + step, max_step)
            if high.iloc[i] > sar_values.iloc[i]:
                trend = 1
                sar_values.iloc[i] = ep
                ep = high.iloc[i]
                af = step

    return sar_values

psar = sar

def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    factor: float = 3.0,
    atr_period: int = 10,
    *,
    length: int | None = None,
    multiplier: float | None = None,
    window: int | None = None,
) -> pd.DataFrame:
    """Supertrend indicator with Pine-style factor and atr_period inputs."""
    atr_period = length if length is not None else atr_period
    atr_period = window if window is not None else atr_period
    factor = multiplier if multiplier is not None else factor
    _validate_length(atr_period, "atr_period")

    atr_values = atr(high, low, close, length=atr_period)
    hl2 = (high + low) / 2
    basic_upper = hl2 + factor * atr_values
    basic_lower = hl2 - factor * atr_values

    upper_band = pd.Series(np.nan, index=close.index, dtype=float)
    lower_band = pd.Series(np.nan, index=close.index, dtype=float)
    supertrend_line = pd.Series(np.nan, index=close.index, dtype=float)
    direction = pd.Series(np.nan, index=close.index, dtype=float)

    for i in range(len(close)):
        if pd.isna(atr_values.iloc[i]):
            continue

        if i == 0 or pd.isna(supertrend_line.iloc[i - 1]):
            upper_band.iloc[i] = basic_upper.iloc[i]
            lower_band.iloc[i] = basic_lower.iloc[i]
            direction.iloc[i] = -1.0
            supertrend_line.iloc[i] = lower_band.iloc[i]
            continue

        prev_upper = upper_band.iloc[i - 1]
        prev_lower = lower_band.iloc[i - 1]
        prev_close = close.iloc[i - 1]
        prev_supertrend = supertrend_line.iloc[i - 1]

        if basic_upper.iloc[i] < prev_upper or prev_close > prev_upper:
            upper_band.iloc[i] = basic_upper.iloc[i]
        else:
            upper_band.iloc[i] = prev_upper

        if basic_lower.iloc[i] > prev_lower or prev_close < prev_lower:
            lower_band.iloc[i] = basic_lower.iloc[i]
        else:
            lower_band.iloc[i] = prev_lower

        if prev_supertrend == prev_upper:
            if close.iloc[i] <= upper_band.iloc[i]:
                supertrend_line.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = 1.0
            else:
                supertrend_line.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = -1.0
        else:
            if close.iloc[i] >= lower_band.iloc[i]:
                supertrend_line.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = -1.0
            else:
                supertrend_line.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = 1.0

    return pd.DataFrame(
        {
            "supertrend": supertrend_line,
            "direction": direction,
            "upper_band": upper_band,
            "lower_band": lower_band,
        }
    )

def chop(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    *,
    window: int | None = None,
) -> pd.Series:
    """Choppiness Index."""
    length = window if window is not None else length
    _validate_length(length)
    atr_sum = true_range(high, low, close).rolling(window=length, min_periods=length).sum()
    high_low_range = (
        high.rolling(window=length, min_periods=length).max()
        - low.rolling(window=length, min_periods=length).min()
    )
    return 100 * np.log10(atr_sum / high_low_range.replace(0, np.nan)) / np.log10(length)

# Momentum oscillators

def rsi(
    prices: pd.Series,
    length: int = 14,
    *,
    window: int | None = None,
) -> pd.Series:
    """Relative Strength Index using Wilder-style smoothing."""
    length = window if window is not None else length
    _validate_length(length)
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = _rma(gain, length)
    avg_loss = _rma(loss, length)

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.mask((avg_gain == 0) & (avg_loss == 0), 50)
    result = result.mask((avg_gain > 0) & (avg_loss == 0), 100)
    result = result.mask((avg_gain == 0) & (avg_loss > 0), 0)
    return result.where(avg_gain.notna())

def macd(
    prices: pd.Series,
    fast_length: int = 12,
    slow_length: int = 26,
    signal_smoothing: int = 9,
    *,
    fast: int | None = None,
    slow: int | None = None,
    signal: int | None = None,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence indicator."""
    fast_length = fast if fast is not None else fast_length
    slow_length = slow if slow is not None else slow_length
    signal_smoothing = signal if signal is not None else signal_smoothing
    _validate_length(fast_length, "fast_length")
    _validate_length(slow_length, "slow_length")
    _validate_length(signal_smoothing, "signal_smoothing")
    if fast_length >= slow_length:
        raise ValueError("fast_length must be less than slow_length.")

    macd_line = ema(prices, length=fast_length, min_periods=fast_length) - ema(
        prices, length=slow_length, min_periods=slow_length
    )
    signal_line = macd_line.ewm(
        span=signal_smoothing, adjust=False, min_periods=signal_smoothing
    ).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
        }
    )

def stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_length: int = 14,
    d_length: int = 3,
    *,
    k_window: int | None = None,
    d_window: int | None = None,
) -> pd.DataFrame:
    """Stochastic oscillator with %K and %D lines."""
    k_length = k_window if k_window is not None else k_length
    d_length = d_window if d_window is not None else d_length
    _validate_length(k_length, "k_length")
    _validate_length(d_length, "d_length")

    lowest_low = low.rolling(window=k_length, min_periods=k_length).min()
    highest_high = high.rolling(window=k_length, min_periods=k_length).max()
    denominator = (highest_high - lowest_low).replace(0, np.nan)
    percent_k = 100 * (close - lowest_low) / denominator
    percent_d = percent_k.rolling(window=d_length, min_periods=d_length).mean()

    return pd.DataFrame({"k": percent_k, "d": percent_d})

stochastic_oscillator = stoch

def roc(
    prices: pd.Series,
    length: int = 20,
    *,
    window: int | None = None,) -> pd.Series:
    """Rate of Change (ROC) indicator."""  
    length = window if window is not None else length
    _validate_length(length)
    return 100 * (prices - prices.shift(length)) / prices.shift(length).replace(0, np.nan)

def momentum(
    prices: pd.Series,
    length: int = 20,
    *,
    window: int | None = None,
) -> pd.Series:
    """Momentum indicator."""
    length = window if window is not None else length
    _validate_length(length)
    return prices - prices.shift(length)

mom = momentum
    

def trix(
    prices: pd.Series,
    length: int = 20,
    *,
    window: int | None = None,
) -> pd.Series:
    """TRIX indicator."""
    length = window if window is not None else length
    _validate_length(length)
    ema1 = ema(prices, length=length)
    ema2 = ema(ema1, length=length)
    ema3 = ema(ema2, length=length)
    trix_values = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1).replace(0, np.nan)
    return trix_values.fillna(0)

def tsi(
    prices: pd.Series,
    long_length: int = 25,
    short_length: int = 13,
    *,
    long_window: int | None = None,
    short_window: int | None = None,
) -> pd.Series:
    """True Strength Index (TSI) indicator."""
    long_length = long_window if long_window is not None else long_length
    short_length = short_window if short_window is not None else short_length
    _validate_length(long_length, "long_length")
    _validate_length(short_length, "short_length")

    momentum = prices - prices.shift(1)
    abs_momentum = momentum.abs()

    ema_momentum_long = ema(momentum, length=long_length)
    ema_momentum_short = ema(ema_momentum_long, length=short_length)

    ema_abs_momentum_long = ema(abs_momentum, length=long_length)
    ema_abs_momentum_short = ema(ema_abs_momentum_long, length=short_length)

    tsi_values = 100 * ema_momentum_short / ema_abs_momentum_short.replace(0, np.nan)
    return tsi_values.fillna(0)

def uo(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length1: int = 7,
    length2: int = 14,
    length3: int = 28,
    weight1: float = 4.0,
    weight2: float = 2.0,
    weight3: float = 1.0,
    *,
    window1: int | None = None,
    window2: int | None = None,
    window3: int | None = None,
) -> pd.Series:
    """Ultimate Oscillator (UO) indicator."""
    length1 = window1 if window1 is not None else length1
    length2 = window2 if window2 is not None else length2
    length3 = window3 if window3 is not None else length3
    _validate_length(length1, "length1")
    _validate_length(length2, "length2")
    _validate_length(length3, "length3")

    prev_close = close.shift(1)
    min_low_or_close = pd.concat([low, prev_close], axis=1).min(axis=1)
    bp = close - min_low_or_close
    tr = true_range(high, low, close)

    avg7 = bp.rolling(window=length1, min_periods=length1).sum() / tr.rolling(window=length1, min_periods=length1).sum().replace(0, np.nan)
    avg14 = bp.rolling(window=length2, min_periods=length2).sum() / tr.rolling(window=length2, min_periods=length2).sum().replace(0, np.nan)
    avg28 = bp.rolling(window=length3, min_periods=length3).sum() / tr.rolling(window=length3, min_periods=length3).sum().replace(0, np.nan)

    uo_values = 100 * (weight1 * avg7 + weight2 * avg14 + weight3 * avg28) / (weight1 + weight2 + weight3)
    return uo_values

ultimate_oscillator = uo

def wpr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    *,
    window: int | None = None,
) -> pd.Series:
    """Williams %R indicator."""
    length = window if window is not None else length
    _validate_length(length)
    highest_high = high.rolling(window=length, min_periods=length).max()
    lowest_low = low.rolling(window=length, min_periods=length).min()
    denominator = (highest_high - lowest_low).replace(0, np.nan)
    return -100 * (highest_high - close) / denominator  

williams_r = wpr

def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 20,
    mult: float = 0.015,
    *,
    window: int | None = None,
    constant: float | None = None,
) -> pd.Series:
    """Commodity Channel Index (CCI) indicator."""
    length = window if window is not None else length
    mult = constant if constant is not None else mult
    _validate_length(length)
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=length, min_periods=length).mean()
    mean_deviation = typical_price.rolling(window=length, min_periods=length).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci_values = (typical_price - sma_tp) / (mult * mean_deviation.replace(0, np.nan))
    return cci_values   

# Bands, channels, and levels

def bb(
    prices: pd.Series,
    length: int = 20,
    mult: float = 2.0,
    *,
    num_std: float | None = None,
    window: int | None = None,
) -> pd.DataFrame:
    """Bollinger Bands with middle, upper, and lower bands."""
    length = window if window is not None else length
    mult = num_std if num_std is not None else mult
    _validate_length(length)
    middle = sma(prices, length=length)
    rolling_std = prices.rolling(window=length, min_periods=length).std(ddof=0)
    upper = middle + mult * rolling_std
    lower = middle - mult * rolling_std

    return pd.DataFrame(
        {
            "middle": middle,
            "upper": upper,
            "lower": lower,
        }
    )

bollinger_bands = bb

def kc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 20,
    mult: float = 1.5,
    *,
    window: int | None = None,
    num_std: float | None = None,
) -> pd.DataFrame:
    """Keltner Channels with middle, upper, and lower bands."""
    length = window if window is not None else length
    mult = num_std if num_std is not None else mult
    _validate_length(length)
    middle = ema(close, length=length)
    atr_values = atr(high, low, close, length=length)
    upper = middle + mult * atr_values
    lower = middle - mult * atr_values

    return pd.DataFrame(
        {
            "middle": middle,
            "upper": upper,
            "lower": lower,
        }
    )

keltner_channel = kc

def dc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 20,
    *,
    window: int | None = None,
) -> pd.DataFrame:
    """Donchian Channels with middle, upper, and lower bands."""
    length = window if window is not None else length
    _validate_length(length)
    upper = high.rolling(window=length, min_periods=length).max()
    lower = low.rolling(window=length, min_periods=length).min()
    middle = (upper + lower) / 2

    return pd.DataFrame(
        {
            "middle": middle,
            "upper": upper,
            "lower": lower,
        }
    )   

donchian_channel = dc

def ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan_length: int = 9,
    kijun_length: int = 26,
    senkou_span_b_length: int = 52,
    displacement: int = 26,
    *,
    tenkan_window: int | None = None,
    kijun_window: int | None = None,
    senkou_span_b_window: int | None = None,
    displacement_window: int | None = None,
) -> pd.DataFrame:
    """Ichimoku Kinko Hyo indicator with Tenkan-sen, Kijun-sen, Senkou Span A/B, and Chikou Span."""
    tenkan_length = tenkan_window if tenkan_window is not None else tenkan_length
    kijun_length = kijun_window if kijun_window is not None else kijun_length
    senkou_span_b_length = senkou_span_b_window if senkou_span_b_window is not None else senkou_span_b_length
    displacement = displacement_window if displacement_window is not None else displacement
    _validate_length(tenkan_length, "tenkan_length")
    _validate_length(kijun_length, "kijun_length")
    _validate_length(senkou_span_b_length, "senkou_span_b_length")
    _validate_length(displacement, "displacement")

    tenkan_sen = (high.rolling(window=tenkan_length, min_periods=tenkan_length).max() + low.rolling(window=tenkan_length, min_periods=tenkan_length).min()) / 2
    kijun_sen = (high.rolling(window=kijun_length, min_periods=kijun_length).max() + low.rolling(window=kijun_length, min_periods=kijun_length).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    senkou_span_b = (
        (
            high.rolling(window=senkou_span_b_length, min_periods=senkou_span_b_length).max()
            + low.rolling(window=senkou_span_b_length, min_periods=senkou_span_b_length).min()
        )
        / 2
    ).shift(displacement)
    chikou_span = close.shift(-displacement)

    return pd.DataFrame(
        {
            "tenkan_sen": tenkan_sen,
            "kijun_sen": kijun_sen,
            "senkou_span_a": senkou_span_a,
            "senkou_span_b": senkou_span_b,
            "chikou_span": chikou_span,
        }
    )

def pivot_points(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.DataFrame:
    """Classic pivot points based on the previous period."""
    pivot = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
    r1 = 2 * pivot - low.shift(1)
    s1 = 2 * pivot - high.shift(1)
    r2 = pivot + (high.shift(1) - low.shift(1))
    s2 = pivot - (high.shift(1) - low.shift(1))
    r3 = high.shift(1) + 2 * (pivot - low.shift(1))
    s3 = low.shift(1) - 2 * (high.shift(1) - pivot)
    return pd.DataFrame({"pivot": pivot, "r1": r1, "s1": s1, "r2": r2, "s2": s2, "r3": r3, "s3": s3})

# Volume indicators

def obv(
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """On-Balance Volume (OBV) indicator."""
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()    

def relative_volume(
    volume: pd.Series,
    length: int = 20,
    *,
    window: int | None = None,
) -> pd.Series:
    """Current volume divided by its rolling average."""
    length = window if window is not None else length
    _validate_length(length)
    average_volume = volume.rolling(window=length, min_periods=length).mean()
    return volume / average_volume.replace(0, np.nan)

RelativeVolume = relative_volume

def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    length: int | None = None,
) -> pd.Series:
    """Volume Weighted Average Price. Uses cumulative VWAP unless length is given."""
    typical_price = (high + low + close) / 3
    pv = typical_price * volume
    if length is None:
        return pv.cumsum() / volume.cumsum().replace(0, np.nan)

    _validate_length(length)
    return pv.rolling(window=length, min_periods=length).sum() / volume.rolling(
        window=length, min_periods=length
    ).sum().replace(0, np.nan)

def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    length: int = 14,
    *,
    window: int | None = None,
) -> pd.Series:
    """Money Flow Index (MFI) indicator."""
    length = window if window is not None else length
    _validate_length(length)
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    positive_flow = money_flow.where(typical_price.diff() > 0, 0.0)
    negative_flow = money_flow.where(typical_price.diff() < 0, 0.0)

    sum_positive_flow = positive_flow.rolling(window=length, min_periods=length).sum()
    sum_negative_flow = negative_flow.rolling(window=length, min_periods=length).sum()

    money_flow_ratio = sum_positive_flow / sum_negative_flow.replace(0, np.nan)
    mfi_values = 100 - (100 / (1 + money_flow_ratio))
    return mfi_values.fillna(100).where(sum_positive_flow.notna())  

def ad(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Accumulation/Distribution Line."""
    high_low_range = (high - low).replace(0, np.nan)
    money_flow_multiplier = ((close - low) - (high - close)) / high_low_range
    money_flow_volume = money_flow_multiplier.fillna(0) * volume
    return money_flow_volume.cumsum()

def cmf(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    length: int = 20,
    *,
    window: int | None = None,
) -> pd.Series:
    """Chaikin Money Flow."""
    length = window if window is not None else length
    _validate_length(length)
    high_low_range = (high - low).replace(0, np.nan)
    money_flow_multiplier = ((close - low) - (high - close)) / high_low_range
    money_flow_volume = money_flow_multiplier.fillna(0) * volume
    return money_flow_volume.rolling(window=length, min_periods=length).sum() / volume.rolling(
        window=length, min_periods=length
    ).sum().replace(0, np.nan)

def adosc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    fast_length: int = 3,
    slow_length: int = 10,
    *,
    fast: int | None = None,
    slow: int | None = None,
) -> pd.Series:
    """Chaikin Oscillator."""
    fast_length = fast if fast is not None else fast_length
    slow_length = slow if slow is not None else slow_length
    _validate_length(fast_length, "fast_length")
    _validate_length(slow_length, "slow_length")
    if fast_length >= slow_length:
        raise ValueError("fast_length must be less than slow_length.")
    ad_line = ad(high, low, close, volume)
    return ema(ad_line, length=fast_length) - ema(ad_line, length=slow_length)

def eom(
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    length: int = 14,
    divisor: float = 100_000_000,
    *,
    window: int | None = None,
) -> pd.Series:
    """Ease of Movement."""
    length = window if window is not None else length
    _validate_length(length)
    _validate_positive(divisor, "divisor")
    distance_moved = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
    box_ratio = (volume / divisor) / (high - low).replace(0, np.nan)
    raw_eom = distance_moved / box_ratio.replace(0, np.nan)
    return sma(raw_eom, length=length)

def pvt(
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Price Volume Trend."""
    pct_change = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    return (pct_change * volume).cumsum()
