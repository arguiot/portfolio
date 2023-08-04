import pandas as pd
import numpy as np
from portfolio_optimization.optimization.GeneralOptimization import GeneralOptimization
from portfolio_optimization.optimization.heuristic import VolatilityOfVolatility
from typing import Type
from .weight_diff import weight_diff


class Portfolio:
    def __init__(
        self,
        base_value: float,
        initial_prices: pd.DataFrame,
        optimiser: Type[GeneralOptimization],
        mcaps: pd.Series = None,
    ):
        self.optimiser = optimiser
        self.weights = pd.Series()
        initial_prices = initial_prices.dropna(axis=1)
        current_prices = initial_prices.iloc[-1]
        # Remove keys from mcaps that are not in initial_prices columns
        if mcaps is not None:
            mcaps = mcaps.reindex(initial_prices.columns)

        self.rebalance(
            initial_prices,
            current_prices,
            base_value,
            mcaps=mcaps,
        )
        # Check that the sum of holdings sum is the base value
        assert np.isclose(self.value(current_prices), base_value), (
            f"Initial value of portfolio is {self.value(current_prices)}, "
            f"but expected value is {base_value}."
        )
        self.latest_optimiser = None

    def assets(self):
        """
        Returns a list of assets in the portfolio.

        Returns:
            list: A list of assets in the portfolio.
        """
        return self.weights.index.tolist()

    def rebalance(
        self,
        df: pd.DataFrame,
        current_prices: pd.Series,
        base_value: float,
        mcaps: pd.Series = None,
    ):
        df = df.dropna(axis=1)
        self.latest_optimiser = self.optimiser(df, mcaps)
        new_weights = self.latest_optimiser.get_weights()

        if self.optimiser is VolatilityOfVolatility:
            new_weights = new_weights.clip(0, 1)

        # update self.weights and calculate the new base value
        self.weights = (
            new_weights  # weight_diff(self.weights, new_weights, applied=True)
        )

        # recalculate self.holdings based on new weights and base value
        self.holdings = self.weights * base_value / pd.Series(current_prices)

    def get_weights(self):
        """
        Returns the weights of the portfolio.

        Returns:
            pd.Series: A pandas Series object representing the weights of each asset in the portfolio.
        """
        return self.weights

    def get_holdings(self):
        """
        Returns the holdings of the portfolio.

        Returns:
            pd.Series: A pandas Series object representing the number of shares of each asset in the portfolio.
        """
        return self.holdings

    def get_metrics(self):
        """
        Returns the metrics of the portfolio.

        Returns:
            pd.Series: A pandas Series object representing the metrics of the portfolio.
        """
        if self.latest_optimiser is None:
            return None
        return self.latest_optimiser.get_metrics()

    def value(self, prices: pd.Series):
        """
        Calculates the current value of the portfolio given a set of asset prices.

        Args:
            prices (pd.Series): A pandas Series object representing the current prices of each asset in the portfolio.

        Returns:
            float: The current value of the portfolio.
        """
        return (self.holdings * prices).sum()

    def apply_yield(self, apy: pd.Series, compounded: bool = True):
        """
        Applies the yield of each asset to the portfolio.

        Args:
            yielded (pd.Series): A pandas Series object representing the yield of each asset in the portfolio. This is the APY that will be converted to daily yield.
            compounded (bool, optional): Whether the yield is compounded or not. Defaults to True.
        """
        # Convert APY to daily yield
        if compounded:
            apy = (1 + apy) ** (1 / 365) - 1
        else:
            apy = apy / 365

        # Set to 0 any undefined yield based on the current holdings
        apy = apy.reindex(self.holdings.index).fillna(0)
        # If yield is negative, set to 0
        apy[apy < 0] = 0

        # Calculate new holdings
        self.holdings = self.holdings * (1 + apy)
