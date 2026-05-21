# Changelog

All notable changes to this project will be documented in this file.

## 0.1.3 - 2026-05-22

- Fixed RSI to use Wilder RMA seeded with the initial SMA.
- Improved Stoch RSI compatibility with TradingView-style RSI values.

## 0.1.1 - 2026-05-22

- Added general indicators: `log_return`, `high_low_range`, `body_ratio`, `relative_volume`, and `hurst_exponent`.
- Added derived indicators: `ema20_ema60_distance` and `bollinger_width`.
- Added compatibility aliases for requested indicator names, including `HistoricalVolatility`, `RelativeVolume`, `HurstExponent`, `EMA20_EMA60_distance`, and `BollingerWidth`.
- Reorganized indicator modules by indicator family.

## 0.1.0 - 2026-05-20

- Added core technical indicators, including moving averages, RSI, MACD, Stochastic, ATR, ADX, Supertrend, Bollinger Bands, Keltner Channel, Donchian Channel, volume indicators, volatility indicators, Ichimoku, SAR, Aroon, and Pivot Points.
- Added derived indicators: `stoch_rsi`, `whale_tracking`, and `spider_chart`.
- Added analytics and portfolio helper functions.
- Added deterministic OHLCV sample data and runnable examples.
- Added tests for public API, return column rules, examples, analytics, and portfolio functions.
