import pandas as pd

from finlib import analytics
import pyta


def test_daily_returns():
    prices = pd.DataFrame({"A": [100.0, 105.0, 110.0]})
    returns = analytics.daily_returns(prices)
    expected = pd.Series([0.05, 0.047619047619047616], index=[1, 2], name="A")
    pd.testing.assert_series_equal(returns["A"], expected)


def test_sharpe_ratio():
    returns = pd.DataFrame({"A": [0.01, 0.02, -0.01]})
    sr = analytics.sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
    assert sr["A"] > 0


def test_max_drawdown():
    returns = pd.DataFrame({"A": [0.1, -0.05, -0.1, 0.05]})
    mdd = analytics.max_drawdown(returns)
    assert mdd["A"] <= 0


def test_pyta_public_alias():
    assert pyta.daily_returns is analytics.daily_returns
