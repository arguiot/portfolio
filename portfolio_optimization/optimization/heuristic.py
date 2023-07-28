from .GeneralOptimization import GeneralOptimization
import numpy as np
import pandas as pd


class Heuristic(GeneralOptimization):
    def __init__(self, df, mcaps=None, callback=lambda df, asset: 1 / df.shape[1]):
        super().__init__(df, mcaps=mcaps)
        self.callback = callback

    def get_weights(self):
        """
        Returns the weights for the assets in the portfolio.

        Returns:
            pd.Series: A pandas Series object containing the weights for the assets in the portfolio.
        """
        weights = pd.Series(name="Weights")
        for asset in self.df.columns:
            weights[asset] = self.callback(self.df, asset)
        return weights

    def get_metrics(self):
        pass


class RiskParity(Heuristic):
    def __init__(self, df, mcaps=None):
        # Risk parity is $$\mathrm{x}_i^{R P}=\frac{1 / \sigma_t^2}{\sum_{i=1}^{\mathrm{N}}\left(1 / \sigma_l^2\right)}, \forall i$$
        def callback(df, asset):
            # calculate the variance of each column
            variance = df[asset].var()
            variances = df.var()
            # calculate the reciprocal (1/variance) of each column
            inverse_variance = 1 / variance
            inverse_variances = 1 / variances
            # calculate the sum of the inverse variances
            total_inverse_variances = np.sum(inverse_variances)
            # calculate the weight of each asset as its inverse variance divided by the total of the inverse variances
            weights = inverse_variance / total_inverse_variances
            # return the weights
            print(weights)
            return weights

        super().__init__(df, callback=callback)
