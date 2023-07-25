from enum import Enum
from .GeneralOptimization import GeneralOptimization
from pypfopt.risk_models import CovarianceShrinkage, sample_cov
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import plotting
import matplotlib.pyplot as plt
import numpy as np


class Markowitz(GeneralOptimization):
    """
    Markowitz portfolio optimization.
    """

    class Mode(Enum):
        SAMPLE_COV = 1
        LEDOIT_WOLF = 2

    def __init__(self, df, cov=None, weight_bounds=(0, 1)):
        """
        Initialize the Markowitz class.

        Parameters:
        -----------
        df : pandas.DataFrame
            A DataFrame of asset prices, where each column represents a different asset.
        """
        super().__init__(df)

        self.weight_bounds = weight_bounds
        self.mode = self.Mode.LEDOIT_WOLF

        if cov is None:
            self.cov_matrix = self.get_cov_matrix()
        else:
            self.cov_matrix = cov

        self.rets = expected_returns.mean_historical_return(df)

    def efficient_frontier(self):
        """
        Clean the optimized weights to ensure they meet any constraints or requirements.

        Returns:
        --------
        None
        """
        ef = EfficientFrontier(
            self.rets,
            self.cov_matrix,
            weight_bounds=self.weight_bounds,
            solver="ECOS_BB",
        )
        return ef

    def get_weights(self):
        """
        Get the optimized portfolio weights.

        Returns:
        --------
        weights : pandas.Series
            A pandas Series object containing the optimized weights for each asset in the portfolio.
        """
        self.ef = self.efficient_frontier()
        self.ef.max_sharpe()
        weights = self.ef.clean_weights()
        return weights

    def get_metrics(self):
        """
        Get the metrics, such as performances, for the optimized portfolio.

        Returns:
        --------
        metrics : dict
            A dictionary containing the metrics for the optimized portfolio.
        """
        metrics = self.ef.portfolio_performance(verbose=False)
        return {
            "apy": metrics[0],
            "annual_volatility": metrics[1],
            "sharpe_ratio": metrics[2],
        }

    def plot_frontier(self):
        """
        Plot the efficient frontier for the portfolio.

        Returns:
        --------
        None
        """
        fig, ax = plt.subplots()
        ef = EfficientFrontier(
            self.rets,
            self.cov_matrix,
            weight_bounds=self.weight_bounds,
            solver="ECOS_BB",
        )
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

    def get_cov_matrix(self):
        """
        Get the covariance matrix for the given data.

        Returns:
        --------
        cov_matrix : pandas.DataFrame
            A pandas DataFrame object containing the covariance matrix for the given data.
        """
        if self.mode == self.Mode.SAMPLE_COV:
            return sample_cov(self.df)
        elif self.mode == self.Mode.LEDOIT_WOLF:
            return CovarianceShrinkage(self.df).ledoit_wolf()
