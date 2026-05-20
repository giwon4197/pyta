import pandas as pd

from finlib import portfolio


def test_portfolio_return():
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.01, -0.01]})
    weights = [0.5, 0.5]
    port = portfolio.portfolio_return(returns, weights)
    expected = pd.Series([0.01, 0.005])
    pd.testing.assert_series_equal(port, expected)


def test_portfolio_volatility():
    returns = pd.DataFrame({"A": [0.01, -0.01], "B": [0.02, -0.02]})
    weights = [0.5, 0.5]
    vol = portfolio.portfolio_volatility(returns, weights, periods_per_year=252)
    assert vol >= 0
