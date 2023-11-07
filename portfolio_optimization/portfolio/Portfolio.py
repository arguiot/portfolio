import pandas as pd
import numpy as np
from ..optimization.GeneralOptimization import GeneralOptimization
from ..optimization.heuristic import VolatilityOfVolatility
from .delegate import PortfolioDelegate
from typing import Type, Callable
from .weight_diff import weight_diff
from typing import Dict


class Portfolio:
    def __init__(
        self,
        base_value: float,
        initial_prices: pd.DataFrame,
        optimiser: Type[GeneralOptimization] | Callable[..., GeneralOptimization],
        mcaps: pd.Series | pd.DataFrame | None = None,
        max_weight: float | Dict[str, float] = 1.0,
        min_weight: float | Dict[str, float] = 0.0,
        **kwargs,  # For additional parameters for the optimiser
    ):
        self.optimiser = optimiser
        self.weights = pd.Series()
        self.delegate = PortfolioDelegate()
        self.raw_weights = pd.Series()
        initial_prices = initial_prices.dropna(axis=1)
        current_prices = initial_prices.iloc[-1]
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.kwargs = kwargs
        self.latest_apy: pd.Series | None = None
        self.base_value = base_value
        # Remove keys from mcaps that are not in initial_prices columns
        if mcaps is not None:
            mcaps = mcaps.reindex(initial_prices.columns)
        local_mcaps = mcaps
        if isinstance(mcaps, pd.DataFrame):
            local_mcaps = mcaps.loc[0]
        self.rebalance(
            initial_prices,
            current_prices,
            base_value,
            mcaps=local_mcaps,
        )
        # Check that the sum of holdings sum is the base value
        assert np.isclose(self.value(current_prices), base_value, 0.01), (
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
        mcaps: pd.Series | None = None,
    ):
        df = df.dropna(axis=1)

        # Check if `max_weight` and `min_weight` are dictionary or not and set to dictionary
        if isinstance(self.max_weight, float):
            max_weight_dict = {key: self.max_weight for key in df.columns}
            max_weight_dict["*"] = self.max_weight  # Using '*' for default value
            self.max_weight = max_weight_dict
        if isinstance(self.min_weight, float):
            min_weight_dict = {key: self.min_weight for key in df.columns}
            min_weight_dict["*"] = self.min_weight  # Using '*' for default value
            self.min_weight = min_weight_dict

        self.latest_optimiser = self.optimiser(df, mcaps)
        self.latest_optimiser.apply_kwargs(self.kwargs)
        setattr(self.latest_optimiser, "latest_apy", self.latest_apy)

        # If optimizer has `weight_bounds` attribute, provide max_weight and min_weight for each asset
        if hasattr(self.latest_optimiser, "weight_bounds"):
            weight_bounds = (self.min_weight["*"], self.max_weight["*"])
            self.latest_optimiser.weight_bounds = weight_bounds
        elif hasattr(self.latest_optimiser, "asset_weight_bounds"):
            asset_weight_bounds = {}
            for asset in df.columns:
                asset_weight_bounds[asset] = (
                    self.min_weight.get(asset, self.min_weight["*"]),
                    self.max_weight.get(asset, self.max_weight["*"]),
                )

            self.latest_optimiser.asset_weight_bounds = asset_weight_bounds

        new_weights = self.latest_optimiser.get_weights()
        self.raw_weights = new_weights.copy()

        # Check and handle weights in case does not meet the 'max_weight'
        while new_weights.max() > self.max_weight["*"]:
            # Get the asset with maximum weight
            max_weight_asset = new_weights.idxmax()

            # Remove the weight not equal to self.max_weight
            new_weights = new_weights[new_weights == self.max_weight["*"]]

            # Set its weight to self.max_weight
            new_weights[max_weight_asset] = self.max_weight["*"]

            # Drop the asset in new_weights from df
            df_rest = df.drop(new_weights.index, axis=1)
            mcaps_rest = None
            if mcaps is not None:
                mcaps_rest = mcaps.drop(new_weights.index, axis=0)

            if df_rest.shape[1] == 1:
                # We set the weight to the remaining weight (1 - sum of weights)
                asset = df_rest.columns[0]
                new_weights[asset] = 1 - new_weights.sum()
                break

            # Re-run the optimiser on the rest of the assets
            local_optimiser = self.optimiser(df_rest, mcaps_rest)
            local_optimiser.apply_kwargs(self.kwargs)
            setattr(local_optimiser, "latest_apy", self.latest_apy)
            new_weights_rest = local_optimiser.get_weights()

            # Normalize the weight so that the sum of weights is 1. We look at the sum of `new_weights`, and we make sure that the sum of `new_weights_rest` is 1 - sum of `new_weights`
            new_weights_rest = new_weights_rest * (1 - new_weights.sum())
            # Combine original weights and the recalculated weights for the remaining assets
            new_weights = pd.concat([new_weights, new_weights_rest])

        if self.optimiser is VolatilityOfVolatility:
            new_weights = new_weights.clip(0, 1)

        assert np.isclose(new_weights.sum(), 1, 0.01), (
            f"Sum of raw weights is {new_weights.sum()}, " f"but expected value is 1."
        )
        real_weights = self.get_current_weights(current_prices)
        self.weights = new_weights

        assert np.isclose(self.weights.sum(), 1, 0.0001), (
            f"Sum of weights is {self.weights.sum()}, " f"but expected value is 1."
        )

        if not hasattr(self, "holdings"):
            self.holdings = base_value * self.weights / current_prices

        new_holdings = self.delegate.rebalance(
            self.holdings, current_prices, self.weights
        )

        # Check that the value of the portfolio is the same as the base value
        assert np.isclose(
            (new_holdings * current_prices).sum(), base_value, 0.01
        ), f"The new holdings value ({(new_holdings * current_prices).sum()}) does not match projected portfolio value ({base_value})"
        self.holdings = new_holdings

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

    def get_current_weights(self, prices: pd.Series):
        """
        Returns the current weights of the portfolio based on the current holdings and prices.

        Returns:
            pd.Series: A pandas Series object representing the current weights of each asset in the portfolio.
        """
        if not hasattr(self, "holdings"):
            return self.weights
        return self.holdings * prices / self.value(prices)

    def match_weights(self, prices: pd.Series):
        """
        Rebalances the portfolio to match the current weights. It adjusts the holdings of each asset to match the current weights, based on the current price and value of the portfolio.

        Returns:
            pd.Series: A pandas Series object representing the current weights of each asset in the portfolio.
        """
        if not hasattr(self, "holdings"):
            return self.weights
        try:
            self.delegate.rebalance(self.holdings, prices, self.weights)
            return self.weights
        except Exception as e:
            print(e)
            return self.weights

    def value(self, prices: pd.Series):
        """
        Calculates the current value of the portfolio given a set of asset prices.

        Args:
            prices (pd.Series): A pandas Series object representing the current prices of each asset in the portfolio.

        Returns:
            float: The current value of the portfolio.
        """
        # if holdings doesn't exist, return 0
        if not hasattr(self, "holdings"):
            return self.base_value
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

        self.latest_apy = apy
        # Calculate new holdings
        self.holdings = self.holdings * (1 + apy)

    def decompose_grouped_tokens(self, rosetta: pd.Series):
        Portfolio.decompose_grouped_tokens(self.holdings, rosetta)

    def decompose_grouped_tokens(current_holdings: pd.Series, rosetta: pd.Series):
        """
        Decomposes the holdings of the portfolio into single assets. This is useful when the portfolio contains LP Tokens representing token pairs.
        """
        # Iterate over the holdings. If a token is present in rosetta, then we decompose it.
        for token in current_holdings.index:
            if token in rosetta.index:
                # Get the holdings of the token
                token_holdings = current_holdings[token]
                # Remove the token from the holdings
                current_holdings = current_holdings.drop(token)
                tokens = rosetta[token]
                for asset in tokens:
                    current_holdings[asset["name"]] = (
                        asset["distribution"] * token_holdings
                    )
        return current_holdings
