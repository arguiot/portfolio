import unittest
import pandas as pd
from portfolio_optimization.portfolio.Portfolio import Portfolio


class TestPortfolio(unittest.TestCase):
    def setUp(self):
        weights = pd.Series({"btc": 0.2, "eth": 0.3, "matic": 0.5})
        base_value = 10000  # Initial investment
        initial_prices = pd.Series({"btc": 50000, "eth": 4000, "matic": 1.2})
        self.portfolio = Portfolio(weights, base_value, initial_prices)

    def test_holdings(self):
        expected_holdings = pd.Series({"btc": 0.04, "eth": 0.75, "matic": 4166.67})
        pd.testing.assert_series_equal(
            self.portfolio.holdings.round(2), expected_holdings
        )

    def test_portfolio_value(self):
        current_prices = pd.Series({"btc": 60000, "eth": 5000, "matic": 1.4})
        expected_value = 60000 * 0.04 + 5000 * 0.75 + 4166.67 * 1.4
        self.assertAlmostEqual(
            self.portfolio.value(current_prices), expected_value, delta=0.01
        )


if __name__ == "__main__":
    unittest.main()
