from .GeneralOptimization import GeneralOptimization
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import expected_returns
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting
from pypfopt import EfficientFrontier, objective_functions
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class BlackLitterman(GeneralOptimization):
    def __init__(self, df, mcaps, views=None, cov=None, weight_bounds=(0, 1)):
        """
        Initialize the BL class.

        Parameters:
        -----------
        df : pandas.DataFrame
            A DataFrame of asset prices, where each column represents a different asset.
        """
        self.weight_bounds = weight_bounds
        super().__init__(df, mcaps)

        if cov is None:
            self.cov_matrix = self.get_cov_matrix()
        else:
            self.cov_matrix = cov

        if views is None:
            self.views = self.compute_expected_returns()
        else:
            self.views = views

        risk_free_asset = "btc" if "btc" in df.columns else df.columns[0]
        self.delta = black_litterman.market_implied_risk_aversion(df[risk_free_asset])

        # Drop all rows in mcaps that are not in cov_matrix
        self.mcaps = self.mcaps[self.cov_matrix.columns]

        self.market_prior = black_litterman.market_implied_prior_returns(
            self.mcaps, self.delta, self.cov_matrix
        )

    def efficient_frontier(self, rets_bl, S_bl):
        """
        Compute the efficient frontier for the given data.

        Parameters:
        -----------
        rets_bl : pandas.Series
            A pandas Series object containing the expected returns for the given data.
        S_bl : pandas.DataFrame
            A pandas DataFrame object containing the covariance matrix for the given data.

        Returns:
        --------
        ef : EfficientFrontier object
            An EfficientFrontier object containing the efficient frontier for the given data.
        """
        self.ef = EfficientFrontier(
            rets_bl,
            S_bl,
            weight_bounds=self.weight_bounds,  # The optimzer is not good at handling constraints, so we set the bounds to (0, 1) and use the `clean_weights` method to remove any assets with zero weight
            solver="ECOS_BB",
        )
        self.ef.add_objective(objective_functions.L2_reg)
        return self.ef

    def get_weights(
        self, risk_free_rate=-0.05
    ):  # Risk free rate is set to -5% by default, which is the rate for the US Dollar during inflation
        bl = BlackLittermanModel(
            self.cov_matrix, pi=self.market_prior, absolute_views=self.views
        )

        self.S_bl = bl.bl_cov()
        self.ret_bl = bl.bl_returns()

        self.ef = self.efficient_frontier(self.ret_bl, self.S_bl)
        risk_free_rate = min(min(self.ef.expected_returns), risk_free_rate)
        self.ef.max_sharpe(risk_free_rate=risk_free_rate)
        weights = self.ef.clean_weights()
        return pd.Series(weights)

    def get_metrics(self):
        """
        Get the metrics, such as performances, for the optimized portfolio.

        Returns:
        --------
        metrics : dict
            A dictionary containing the metrics for the optimized portfolio.
        """
        if self.ef.weights is None:
            return None
        metrics = self.ef.portfolio_performance(verbose=False)
        return {
            "apy": metrics[0],
            "annual_volatility": metrics[1],
            "sharpe_ratio": metrics[2],
        }

    def get_cov_matrix(self):
        """
        Get the covariance matrix for the given data.

        Returns:
        --------
        cov_matrix : pandas.DataFrame
            A pandas DataFrame object containing the covariance matrix for the given data.
        """
        return CovarianceShrinkage(self.df).ledoit_wolf()

    def compute_expected_returns(self):
        """
        Compute the expected returns for the given data.

        Returns:
        --------
        expected_returns : pandas.Series
            A pandas Series object containing the expected returns for the given data.
        """
        return expected_returns.mean_historical_return(self.df, compounding=False)

    def plot_frontier(self):
        """
        Plot the efficient frontier for the portfolio.

        Returns:
        --------
        None
        """
        fig, ax = plt.subplots()
        ef = self.efficient_frontier(self.ret_bl, self.S_bl)
        ef_max_sharpe = ef.deepcopy()
        plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

        # Find the tangency portfolio
        ef_max_sharpe.max_sharpe()
        ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
        ax.scatter(
            std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe"
        )

        # Generate random portfolios
        n_samples = 15000
        w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
        rets = w.dot(ef.expected_returns)
        stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
        sharpes = rets / stds
        ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

        # Output
        ax.set_title("Efficient Frontier with random portfolios")
        ax.legend()
        plt.tight_layout()
        plt.show()
