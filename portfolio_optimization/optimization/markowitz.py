from enum import Enum
from .GeneralOptimization import GeneralOptimization
from pypfopt.risk_models import risk_matrix
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import expected_returns
from pypfopt import plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Markowitz(GeneralOptimization):
    """
    Markowitz portfolio optimization.
    """

    class CovMode(Enum):
        SAMPLE_COV = "sample_cov"
        SEMICOVARIANCE = "semicovariance"
        EXP_COV = "exp_cov"
        LEDOIT_WOLF = "ledoit_wolf"
        LEDOIT_WOLF_CONSTANT_VARIANCE = "ledoit_wolf_constant_variance"
        LEDOIT_WOLF_SINGLE_FACTOR = "ledoit_wolf_single_factor"
        LEDOIT_WOLF_CONSTANT_CORRELATION = "ledoit_wolf_constant_correlation"
        ORACLE_APPROXIMATING = "oracle_approximating"

    class EfficientPortfolio(Enum):
        MAX_SHARPE = "max_sharpe"
        MIN_VOLATILITY = "min_volatility"
        MAX_RETURN = "max_return"

    def __init__(self, df, mcaps=None, cov=None, weight_bounds=(0, 1)):
        """
        Initialize the Markowitz class.

        Parameters:
        -----------
        df : pandas.DataFrame
            A DataFrame of asset prices, where each column represents a different asset.
        """
        super().__init__(df, mcaps=mcaps)

        self.weight_bounds = weight_bounds
        self.mode = self.CovMode.LEDOIT_WOLF
        self.efficient_portfolio = self.EfficientPortfolio.MAX_SHARPE

        self.cov_matrix = cov
        self.rets = None

    def efficient_frontier(self):
        """
        Compute the efficient frontier for the portfolio.

        Returns:
        --------
        ef : EfficientFrontier object
            An EfficientFrontier object containing the efficient frontier for the portfolio.
        """
        ef = EfficientFrontier(
            self.rets,
            self.cov_matrix,
            weight_bounds=self.weight_bounds,
            solver="ECOS_BB",
        )
        return ef

    def get_weights(
        self,
        risk_free_rate=None,
    ):  # Risk free rate is set to -5% by default, which is the rate for the US Dollar during inflation
        """
        Get the optimized portfolio weights.

        Returns:
        --------
        weights : pandas.Series
            A pandas Series object containing the optimized weights for each asset in the portfolio.
        """
        self.delegate.setup(self)
        if self.cov_matrix is None:
            self.cov_matrix = self.get_cov_matrix()
        if self.rets is None:
            self.rets = expected_returns.mean_historical_return(
                self.df, log_returns=True
            )
        self.ef = self.efficient_frontier()
        risk_free_rate = (
            min(self.ef.expected_returns) if risk_free_rate is None else risk_free_rate
        )
        assert max(self.ef.expected_returns) >= risk_free_rate, (
            "Expected returns are less than the risk free rate. "
            "This is not possible. Please check your data."
        )
        if isinstance(self.efficient_portfolio, self.EfficientPortfolio):
            if self.efficient_portfolio == self.EfficientPortfolio.MAX_SHARPE:
                self.ef.max_sharpe(risk_free_rate=risk_free_rate)
            elif self.efficient_portfolio == self.EfficientPortfolio.MIN_VOLATILITY:
                self.ef.min_volatility()
            elif self.efficient_portfolio == self.EfficientPortfolio.MAX_RETURN:
                self.ef.max_quadratic_utility()
        elif isinstance(self.efficient_portfolio, float):
            self.ef.efficient_return(self.efficient_portfolio)

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
        return risk_matrix(self.df, method=self.mode.value)
