from __future__ import annotations

import pyta
from sample_data import make_sample_ohlcv


def main() -> None:
    df = make_sample_ohlcv()

    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]

    macd = pyta.macd(close)
    supertrend = pyta.supertrend(high, low, close, 3, 14)
    stoch_rsi = pyta.stoch_rsi(close)
    spider = pyta.spider_chart(high, low, close, volume)
    whale = pyta.whale_tracking(open_, high, low, close, volume)

    print("Sample OHLCV")
    print(df.tail(3))
    print()

    print("MACD")
    print(macd.tail(3))
    print()

    print("Supertrend")
    print(supertrend.tail(3))
    print()

    print("Stoch RSI")
    print(stoch_rsi[["k", "d"]].tail(3))
    print()

    print("Spider Chart")
    print(spider[["avg_score", "long_entry", "short_entry"]].tail(3))
    print()

    print("Whale Tracking")
    print(whale[["normalized_weight", "institution_pressure"]].tail(3))


if __name__ == "__main__":
    main()
