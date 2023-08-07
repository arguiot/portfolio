from .GeneralOptimization import GeneralOptimization
import numpy as np
import pandas as pd
from pypfopt import expected_returns


class Heuristic(GeneralOptimization):
    def __init__(
        self,
        df,
        mcaps=None,
        pre_processing=lambda df: df.shape[1],
        post_processing=lambda df, asset, pre_result: 1 / pre_result,
    ):
        super().__init__(df, mcaps=mcaps)
        self.pre_processing = pre_processing
        self.post_processing = post_processing

    def get_weights(self):
        """
        Returns the weights for the assets in the portfolio.

        Returns:
            pd.Series: A pandas Series object containing the weights for the assets in the portfolio.
        """
        weights = pd.Series(name="Weights")
        partial_result = self.pre_processing(self.df)
        for asset in self.df.columns:
            weights[asset] = self.post_processing(self.df, asset, partial_result)

        # Normalize weights to sum to 1
        weights = weights.fillna(0)
        weights /= weights.sum()
        return weights

    def get_metrics(self):
        pass


class FastRiskParity(Heuristic):
    def __init__(self, df, mcaps=None):
        # Use the apply function to calculate log returns
        self.log_returns = expected_returns.returns_from_prices(df, log_returns=True)

        # Risk parity is $$\mathrm{x}_i^{R P}=\frac{1 / \sigma_t^2}{\sum_{i=1}^{\mathrm{N}}\left(1 / \sigma_l^2\right)}, \forall i$$
        def callback(df, asset, pre_result):
            # calculate the variance of each column
            variance = self.log_returns[asset].var()
            variances = self.log_returns.var()
            # calculate the reciprocal (1/variance) of each column
            inverse_variance = 1 / variance
            inverse_variances = 1 / variances
            # calculate the sum of the inverse variances
            total_inverse_variances = np.sum(inverse_variances)
            # calculate the weight of each asset as its inverse variance divided by the total of the inverse variances
            weights = inverse_variance / total_inverse_variances
            # return the weights
            return weights

        super().__init__(df, post_processing=callback)


class RewardToRisk(Heuristic):
    def __init__(self, df, mcaps=None):
        # Use the apply function to calculate log returns
        self.log_returns = expected_returns.returns_from_prices(df, log_returns=True)

        # Reward-to-risk is weight of the asset relative to its risk, measured as standard deviation of returns.
        def callback(df, asset, pre_result):
            # Expected return (mean) of each asset
            mu = self.log_returns[asset].mean()

            # Variance of each asset
            variance = self.log_returns[asset].var()

            # Avoid division by zero
            if variance == 0:
                variance = np.finfo(float).eps

            # Compute weight for the current asset
            weight = mu / variance

            # Compute similar measure for all assets
            mu_all = self.log_returns.mean()
            variance_all = self.log_returns.var()

            # Avoid division by zero
            variance_all[variance_all == 0] = np.finfo(float).eps

            weights_all = mu_all / variance_all

            # Normalization factor, sum of all weights
            total_weights = np.sum(weights_all)

            # Normalize weight of current asset
            weight /= total_weights

            return max(weight, 0)

        super().__init__(df, mcaps, post_processing=callback)


class ValueAtRisk(Heuristic):
    def __init__(self, df, mcaps=None):
        # Use the apply function to calculate log returns
        self.log_returns = expected_returns.returns_from_prices(df, log_returns=True)

        # Value-at-risk is $$\mathrm{x}_i^{\mathrm{VaRH}}=\frac{1 / \mathrm{VaR}_{99 \%, i}}{\sum_{i=1}^{\mathrm{N}}\left(1 / \operatorname{VaR}_{99 \%, i}\right)}, \forall i$$
        def callback(df, asset, pre_result):
            # Calculate the 99% VaR
            var = self.log_returns[asset].quantile(0.01)
            vars = self.log_returns.quantile(0.01)
            # Calculate the reciprocal (1/variance) of each column
            inverse_var = 1 / var
            inverse_vars = 1 / vars
            # Calculate the sum of the inverse variances
            total_inverse_vars = np.sum(inverse_vars)
            # Calculate the weight of each asset as its inverse variance divided by the total of the inverse variances
            weights = inverse_var / total_inverse_vars
            # Return the weights
            return weights

        super().__init__(df, mcaps, post_processing=callback)


class RewardToVaR(Heuristic):
    def __init__(self, df, mcaps=None):
        # Use the apply function to calculate log returns
        self.log_returns = expected_returns.returns_from_prices(df, log_returns=True)

        # Reward-to-risk is weight of the asset relative to its risk, measured as standard deviation of returns.
        def callback(df, asset, pre_result):
            # Expected return (mean) of each asset
            mu = self.log_returns[asset].mean()

            # Variance of each asset
            var = self.log_returns[asset].quantile(0.01)

            # Avoid division by zero
            if var == 0:
                var = np.finfo(float).eps

            # Compute weight for the current asset
            weight = mu / var

            # Compute similar measure for all assets
            mu_all = self.log_returns.mean()
            var_all = self.log_returns.quantile(0.01)

            # Avoid division by zero
            var_all[var_all == 0] = np.finfo(float).eps

            weights_all = mu_all / var_all

            # Normalization factor, sum of all weights
            total_weights = np.sum(weights_all)

            # Normalize weight of current asset
            weight /= total_weights

            return max(weight, 0)

        super().__init__(df, mcaps, post_processing=callback)


class Combination(Heuristic):
    def __init__(self, df, mcaps=None):
        # We use a combination of the Default Heuristic, Fast Risk Parity, Reward To Risk, Value At Risk and Reward To VaR
        self.default = Heuristic(df, mcaps)
        self.fast_risk_parity = FastRiskParity(df, mcaps)
        self.reward_to_risk = RewardToRisk(df, mcaps)
        self.value_at_risk = ValueAtRisk(df, mcaps)
        self.reward_to_var = RewardToVaR(df, mcaps)

        def pre_processing(df):
            return (
                self.default.get_weights(),
                self.fast_risk_parity.get_weights(),
                self.reward_to_risk.get_weights(),
                self.value_at_risk.get_weights(),
                self.reward_to_var.get_weights(),
            )

        def post_processing(df, asset, pre_result):
            (
                default,
                fast_risk_parity,
                reward_to_risk,
                value_at_risk,
                reward_to_var,
            ) = pre_result
            # Average of the weights
            weight = (
                default[asset]
                + fast_risk_parity[asset]
                + reward_to_risk[asset]
                + value_at_risk[asset]
                + reward_to_var[asset]
            ) / 5
            return weight

        super().__init__(
            df, mcaps, pre_processing=pre_processing, post_processing=post_processing
        )


class VolatilityOfVolatility(Heuristic):
    def __init__(self, df, mcaps=None):
        def pre_processing(df):
            # Use the apply function to calculate log returns
            log_returns = expected_returns.returns_from_prices(df, log_returns=True)
            window = log_returns.rolling(window=30, min_periods=3)
            # 1-month rolling volatility
            vol = window.std()
            # Volatility of volatility
            vol_of_vol = vol.rolling(window=30).std()
            return vol, vol_of_vol

        def post_processing(df, asset, pre_result):
            # Volatility of the asset for the last month
            vol, vol_of_vol = pre_result
            vol_asset = vol[asset][-1]
            vol_of_vol_asset = vol_of_vol[asset][-1]
            local_total_vol = vol_asset + vol_of_vol_asset
            global_total_vol = vol[asset].sum() + vol_of_vol[asset].sum()

            if global_total_vol == 0 or local_total_vol == 0:
                return 0
            # Weigth
            weight = (1 / local_total_vol) / (1 / global_total_vol)
            return weight

        super().__init__(
            df, mcaps, pre_processing=pre_processing, post_processing=post_processing
        )


class SimpleVolatility(Heuristic):
    def __init__(self, df, mcaps=None):
        def pre_processing(df):
            # Use the apply function to calculate log returns
            log_returns = expected_returns.returns_from_prices(df, log_returns=True)
            window = log_returns.rolling(window=30, min_periods=3)
            # 1-month rolling volatility
            vol = window.std()
            return vol

        def post_processing(df, asset, pre_result):
            # Volatility of the asset for the last month
            vol = pre_result
            vol_asset = vol[asset][-1]
            global_total_vol = vol[asset].sum()

            if global_total_vol == 0 or vol_asset == 0:
                return 0
            # Weigth
            weight = (1 / vol_asset) / (1 / global_total_vol)
            return weight

        super().__init__(
            df, mcaps, pre_processing=pre_processing, post_processing=post_processing
        )
