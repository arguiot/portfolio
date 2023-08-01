import pandas as pd
import numpy as np
from portfolio_optimization.optimization.GeneralOptimization import GeneralOptimization
from portfolio_optimization.optimization.heuristic import VolatilityOfVolatility
from typing import Type
from .weight_diff import weight_diff


class Portfolio:
    """
    A class representing a portfolio of assets with a given set of weights.

    Attributes:
        weights (pd.Series): A pandas Series object representing the weights of each asset in the portfolio.
        base_value (float): The initial value of the portfolio.
        holdings (pd.Series): A pandas Series object representing the number of shares of each asset in the portfolio.
    """

    def __init__(
        self,
        weights: pd.Series,
        base_value: float,
        initial_prices: pd.Series,
        optimiser: Type[GeneralOptimization],
    ):
        self.optimiser = optimiser
        self.weights = pd.Series(weights)
        self.holdings = self.weights * base_value / pd.Series(initial_prices)
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
        self.latest_optimiser = self.optimiser(df, mcaps)
        new_weights = self.latest_optimiser.get_weights()

        if self.optimiser is VolatilityOfVolatility:
            new_weights = new_weights.clip(0, 1)

        # update self.weights and calculate the new base value
        self.weights = weight_diff(self.weights, new_weights, applied=True)

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
