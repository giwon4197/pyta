# pyta

`pyta`는 Python에서 금융 데이터와 기술적 지표를 계산하기 위한 가벼운 라이브러리입니다.
TradingView Pine Script를 쓰던 사람이 Python에서도 비슷한 감각으로 지표를 사용할 수 있게 만드는 것을 목표로 합니다.

기존 내부 패키지 이름인 `finlib`도 남아 있지만, 새 코드에서는 `pyta`로 사용하는 것을 권장합니다.

## 설치

개발 중에는 프로젝트 루트에서 editable 모드로 설치합니다.

```bash
python -m pip install -e .
```

테스트를 실행하려면:

```bash
python -m pytest
```

## 기본 사용법

```python
import pandas as pd
import pyta

close = pd.Series([100, 102, 101, 105, 108, 107, 110], dtype=float)
high = close + 2
low = close - 2
volume = pd.Series([1000, 1200, 900, 1500, 1800, 1300, 1600], dtype=float)

rsi = pyta.rsi(close, length=14)
macd = pyta.macd(close, 12, 26, 9)
adx = pyta.adx(high, low, close, length=14)
vwap = pyta.vwap(high, low, close, volume)
```

Pine Script처럼 짧게 쓰는 것도 지원합니다.

```python
pyta.supertrend(high, low, close, 3, 14)
pyta.bb(close, 20, 2)
pyta.stoch_rsi(close, 3, 3, 14, 14)
```

## 파일 구조

```text
src/finlib/indicators.py
  기본 지표와 표준 기술적 지표

src/finlib/derived_indicators.py
  여러 기본 지표와 조건을 조합한 파생지표

src/finlib/analytics.py
  수익률, 변동성, 샤프 비율, 최대 낙폭 같은 분석 함수

src/finlib/portfolio.py
  포트폴리오 수익률과 변동성 함수
```

사용자는 파일 위치를 몰라도 대부분 `pyta.함수명(...)` 형태로 바로 사용할 수 있습니다.

## 기본 지표

### 이동평균

| 함수 | 설명 |
| --- | --- |
| `sma` | 단순 이동평균 |
| `ema` | 지수 이동평균 |
| `wma` | 가중 이동평균 |
| `vwma` | 거래량 가중 이동평균 |
| `hma` | Hull 이동평균 |
| `swma` | Smoothed 이동평균 |

### 모멘텀 / 오실레이터

| 함수 | 설명 |
| --- | --- |
| `rsi` | Relative Strength Index |
| `macd` | MACD, signal, histogram |
| `stoch` / `stochastic_oscillator` | Stochastic oscillator |
| `wpr` / `williams_r` | Williams %R |
| `mfi` | Money Flow Index |
| `cci` | Commodity Channel Index |
| `roc` | Rate of Change |
| `momentum` / `mom` | Momentum |
| `trix` | Triple EMA momentum |
| `tsi` | True Strength Index |
| `uo` / `ultimate_oscillator` | Ultimate Oscillator |

### 추세 / 채널 / 변동성

| 함수 | 설명 |
| --- | --- |
| `atr` | Average True Range |
| `natr` | Normalized ATR |
| `adx` | ADX, +DI, -DI |
| `supertrend` | Supertrend |
| `bb` / `bollinger_bands` | Bollinger Bands |
| `kc` / `keltner_channel` | Keltner Channel |
| `dc` / `donchian_channel` | Donchian Channel |
| `aroon` | Aroon Up/Down |
| `sar` / `psar` | Parabolic SAR |
| `ichimoku` | Ichimoku Cloud |
| `pivot_points` | Classic pivot points |
| `stddev` / `stdev` | Rolling standard deviation |
| `variance` | Rolling variance |
| `historical_volatility` | Annualized historical volatility |
| `chop` | Choppiness Index |

### 거래량

| 함수 | 설명 |
| --- | --- |
| `obv` | On-Balance Volume |
| `vwap` | Volume Weighted Average Price |
| `ad` | Accumulation/Distribution Line |
| `cmf` | Chaikin Money Flow |
| `adosc` | Chaikin Oscillator |
| `eom` | Ease of Movement |
| `pvt` | Price Volume Trend |

## 파생지표

파생지표는 여러 기본 지표, 가격 위치, 거래량 조건, 신호 조건을 조합해서 만든 지표입니다.
이 함수들은 `src/finlib/derived_indicators.py`에 둡니다.

| 함수 | 설명 |
| --- | --- |
| `stoch_rsi` | RSI를 Stochastic으로 변환한 파생 오실레이터 |
| `whale_tracking` | 거래대금, 가격 위치, 변동성 조건을 조합한 매집/분산 추적 지표 |
| `institution_pressure_normalized_weight` | `whale_tracking`의 호환 이름 |
| `spider_chart` / `spider_chart_indicator` | 여러 오실레이터를 0~1 점수로 정규화해 평균낸 파생 점수 지표 |

예시:

```python
stoch_rsi = pyta.stoch_rsi(close, 3, 3, 14, 14)
spider = pyta.spider_chart(high, low, close, volume)
whale = pyta.whale_tracking(open_, high, low, close, volume)
```

## 반환 컬럼 규칙

여러 값을 반환하는 함수는 `pandas.DataFrame`을 반환합니다.

| 함수 | 컬럼 |
| --- | --- |
| `macd` | `macd`, `signal`, `histogram` |
| `stoch` | `k`, `d` |
| `adx` | `plus_di`, `minus_di`, `adx` |
| `bb` | `middle`, `upper`, `lower` |
| `kc` | `middle`, `upper`, `lower` |
| `dc` | `middle`, `upper`, `lower` |
| `supertrend` | `supertrend`, `direction`, `upper_band`, `lower_band` |
| `aroon` | `aroon_up`, `aroon_down` |
| `ichimoku` | `tenkan_sen`, `kijun_sen`, `senkou_span_a`, `senkou_span_b`, `chikou_span` |
| `stoch_rsi` | `k`, `d`, `rsi`, `stoch_rsi` |
| `spider_chart_indicator` | `avg_score`, `rsi`, `stoch`, `cor`, `mfi`, `wpr`, `pr`, `cmo`, `aos`, `long_entry`, `short_entry`, `long_stop`, `short_stop` |

## 입력 규칙

일반적으로 필요한 Series를 직접 넘깁니다.

```python
pyta.rsi(close)
pyta.adx(high, low, close)
pyta.vwap(high, low, close, volume)
pyta.whale_tracking(open_, high, low, close, volume)
```

`open`은 Python 내장 함수 이름이므로 변수명은 `open_`처럼 쓰는 것을 권장합니다.

기간 인자는 Pine Script 감각에 맞춰 대부분 `length`를 사용합니다.
일부 함수는 짧은 위치 인자도 지원합니다.

```python
pyta.rsi(close, length=14)
pyta.macd(close, 12, 26, 9)
pyta.supertrend(high, low, close, 3, 14)
```

## 함수 이름 규칙

대표 이름은 Pine Script에서 익숙한 짧은 이름을 우선합니다.
다만 읽기 쉬운 풀네임도 alias로 같이 제공합니다.

| 대표 이름 | Alias |
| --- | --- |
| `bb` | `bollinger_bands` |
| `kc` | `keltner_channel` |
| `dc` | `donchian_channel` |
| `stoch` | `stochastic_oscillator` |
| `wpr` | `williams_r` |
| `uo` | `ultimate_oscillator` |
| `sar` | `psar` |
| `stddev` | `stdev` |
| `momentum` | `mom` |
| `whale_tracking` | `institution_pressure_normalized_weight` |
| `spider_chart` | `spider_chart_indicator` |

새 함수는 기본적으로 이 규칙을 따릅니다.

```text
기본 지표: 짧은 Pine식 이름 우선, 풀네임 alias 제공
파생지표: 의미가 드러나는 snake_case 이름 사용, 기존 이름은 alias로 보존
```

## 분석 / 포트폴리오 함수

| 함수 | 설명 |
| --- | --- |
| `daily_returns` | 가격 데이터에서 일간 수익률 계산 |
| `cumulative_returns` | 누적 수익률 계산 |
| `annualized_volatility` | 연율화 변동성 |
| `sharpe_ratio` | 샤프 비율 |
| `max_drawdown` | 최대 낙폭 |
| `portfolio_return` | 포트폴리오 수익률 |
| `portfolio_volatility` | 포트폴리오 변동성 |

## 샘플 데이터 / 예제

예제용 OHLCV 데이터 생성기는 `examples/sample_data.py`에 있습니다.
랜덤 노이즈를 쓰지만 seed가 고정되어 같은 입력이면 항상 같은 데이터가 생성됩니다.

```python
from examples.sample_data import make_sample_ohlcv

df = make_sample_ohlcv(rows=240, seed=42)
```

기본 지표와 파생지표를 한 번에 확인하려면:

```bash
python examples/basic_usage.py
```

## 배포 준비

개발 의존성까지 설치하려면:

```bash
python -m pip install -e ".[dev]"
```

패키지 빌드:

```bash
python -m build
```

빌드 산출물 검사:

```bash
python -m twine check dist/*
```

PyPI에 올리기 전에는 `pyproject.toml`의 `project.urls` 값을 실제 저장소와 문서 주소로 바꾸는 것을 권장합니다.
