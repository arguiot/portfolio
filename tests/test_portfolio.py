import unittest
from unittest.mock import create_autospec, patch
import pandas as pd
from portfolio_optimization.portfolio.Portfolio import Portfolio
from portfolio_optimization.optimization import GeneralOptimization, Markowitz


class TestPortfolio(unittest.TestCase):
    def create_portfolio(self, weights, base_value, initial_prices, optimiser):
        return Portfolio(weights, base_value, initial_prices, optimiser)

    def test_holdings(self):
        weights = pd.Series({"btc": 0.2, "eth": 0.3, "matic": 0.5})
        base_value = 10000  # Initial investment
        initial_prices = pd.Series({"btc": 50000, "eth": 4000, "matic": 1.2})
        portfolio = self.create_portfolio(
            weights, base_value, initial_prices, GeneralOptimization
        )

        expected_holdings = pd.Series({"btc": 0.04, "eth": 0.75, "matic": 4166.67})
        pd.testing.assert_series_equal(portfolio.holdings.round(2), expected_holdings)

    def test_portfolio_value(self):
        weights = pd.Series({"btc": 0.2, "eth": 0.3, "matic": 0.5})
        base_value = 10000  # Initial investment
        initial_prices = pd.Series({"btc": 50000, "eth": 4000, "matic": 1.2})
        portfolio = self.create_portfolio(
            weights, base_value, initial_prices, GeneralOptimization
        )

        current_prices = pd.Series({"btc": 60000, "eth": 5000, "matic": 1.4})
        expected_value = 60000 * 0.04 + 5000 * 0.75 + 4166.67 * 1.4
        self.assertAlmostEqual(
            portfolio.value(current_prices), expected_value, delta=0.01
        )

    def test_portfolio_rebalance(self):
        weights = pd.Series({"btc": 0.2, "eth": 0.3, "matic": 0.5})
        base_value = 10000  # Initial investment
        initial_prices = pd.Series({"btc": 50000, "eth": 4000, "matic": 1.2})

        # create a MockOptimizer class that inherits from GeneralOptimization
        class MockOptimizer(GeneralOptimization):
            # implement the abstract methods
            def get_weights(self):
                return pd.Series({"btc": 0.3, "eth": 0.3, "matic": 0.4})

            def get_metrics(self):
                pass  # return whatever is appropriate for your use case

        portfolio = self.create_portfolio(
            weights, base_value, initial_prices, MockOptimizer
        )

        df = pd.DataFrame()  # DataFrame used for rebalancing

        portfolio.rebalance(df, initial_prices, base_value)

        # Check that new weights have been set
        expected_weights = pd.Series({"btc": 0.3, "eth": 0.3, "matic": 0.4})
        pd.testing.assert_series_equal(portfolio.weights, expected_weights)

    def test_portfolio_value_over_time(self):
        # Day 1: 100% BTC, BTC = 100$, so portfolio value is 100$
        weights = pd.Series({"btc": 1.0})
        base_value = 100  # Initial investment
        initial_prices = pd.Series({"btc": 100})

        # create a MockOptimizer class that inherits from GeneralOptimization
        class MockOptimizerDay3(GeneralOptimization):
            # implement the abstract methods
            def get_weights(self):
                return pd.Series({"btc": 0.5, "eth": 0.5})

            def get_metrics(self):
                pass  # return whatever is appropriate for your use case

        portfolio = self.create_portfolio(
            weights, base_value, initial_prices, GeneralOptimization
        )

        # Day 2: BTC = 200$, so portfolio value is 200$
        current_prices_day2 = pd.Series({"btc": 200})
        self.assertAlmostEqual(portfolio.value(current_prices_day2), 200, delta=0.01)

        # Day 3 (rebalancing): 50% BTC, 50% ETH, BTC = 200$ ETH = 150$, so portfolio value is 200$ because we just rebalanced
        day3_prices = pd.Series({"btc": 200, "eth": 150})
        portfolio = self.create_portfolio(
            pd.Series({"btc": 1.0, "eth": 0}),
            200,
            day3_prices,
            MockOptimizerDay3,
        )

        df = pd.DataFrame()  # DataFrame used for rebalancing
        portfolio.rebalance(df, day3_prices, 200)

        self.assertAlmostEqual(
            portfolio.value(pd.Series({"btc": 200, "eth": 150})), 200, delta=0.01
        )

        # Day 4: BTC = 200$, ETH = 300$, so portfolio value is 300$
        self.assertAlmostEqual(
            portfolio.value(pd.Series({"btc": 200, "eth": 300})), 300, delta=0.01
        )

        self.assertAlmostEqual(portfolio.holdings["eth"], 2 / 3, delta=0.01)


if __name__ == "__main__":
    unittest.main()
