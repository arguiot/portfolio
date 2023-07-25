import pandas as pd
import numpy as np
from portfolio_optimization.optimization.GeneralOptimization import GeneralOptimization
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
        self.base_value = base_value
        self.holdings = self.weights * self.base_value / pd.Series(initial_prices)

    def assets(self):
        """
        Returns a list of assets in the portfolio.

        Returns:
            list: A list of assets in the portfolio.
        """
        return self.weights.index.tolist()

    def rebalance(self, df: pd.DataFrame):
        new_weights = self.optimiser(df).get_weights()
        self.weights = weight_diff(self.weights, new_weights, applied=True)

    def value(self, prices: pd.Series):
        """
        Calculates the current value of the portfolio given a set of asset prices.

        Args:
            prices (pd.Series): A pandas Series object representing the current prices of each asset in the portfolio.

        Returns:
            float: The current value of the portfolio.
        """
        return (self.holdings * prices).sum()
