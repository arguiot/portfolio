import pandas as pd


class PortfolioDelegate:
    def rebalance(
        self,
        holdings: pd.Series,
        prices: pd.Series,
        target_weights: pd.Series,
        base_value: float,
    ) -> pd.Series:
        """
        Rebalance the portfolio to the target weights. The portfolio is rebalanced by
        buying or selling assets. The amount of assets to buy or sell is determined by
        the difference between the current and target weights.

        Parameters
        ----------
        holdings : pd.Series
            The current holdings of the portfolio.
        prices : pd.Series
            The current prices of the assets.
        target_weights : pd.Series
            The target weights of the assets.
        """
        value = (holdings * prices).sum()
        new_holdings = value * target_weights / prices
        return new_holdings
