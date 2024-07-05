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

    def __init__(self, df, mcaps=None, cov=None, max_weights=None):
        """
        Initialize the Markowitz class.

        Parameters:
        -----------
        df : pandas.DataFrame
            A DataFrame of asset prices, where each column represents a different asset.
        mcaps : dict, optional
            A dictionary of market capitalizations for each asset.
        cov : pandas.DataFrame, optional
            A pre-computed covariance matrix.
        max_weights : dict, optional
            A dictionary specifying maximum weights for assets or asset classes.
        """
        super().__init__(df, mcaps=mcaps)

        self.mode = self.CovMode.LEDOIT_WOLF
        self.efficient_portfolio = self.EfficientPortfolio.MAX_SHARPE

        self.cov_matrix = cov
        self.rets = None
        self.max_weights = max_weights or {}
        self.ef = None

    def process_constraints(self):
        print("\nApplying constraints:")
        sector_mapper = {}
        sector_upper = {}
        weight_bounds = [(0, 1)] * len(self.df.columns)

        print(f"Max Weight: {self.max_weights}")

        default_max = self.max_weights.get("*", 1.0)

        # Set default weight bounds for assets not explicitly constrained
        for asset in self.df.columns:
            index = self.df.columns.get_loc(asset)
            if weight_bounds[index] != (0, 1):
                continue
            weight_bounds[index] = (0, default_max)
            print(f"Setting default weight bound for {asset}: (0, {default_max})")

        for key, value in self.max_weights.items():
            print(f"Processing constraint for {key}, with value {value}")
            if isinstance(value, (int, float)):
                if key == "*":
                    continue  # We'll apply the default constraint later
                elif key.lower() in self.df.columns:
                    index = self.df.columns.get_loc(key.lower())
                    weight_bounds[index] = (0, value)
                    print(
                        f"Setting weight bound for {key} at index {index}: (0, {value})"
                    )
                else:
                    print(
                        f"Warning: Asset {key} not found in data, skipping constraint"
                    )
            elif isinstance(value, dict):
                class_name = key
                print(f"Adding sector constraint for {class_name}:")
                for asset in value.get("assets", []):
                    if asset.lower() in self.df.columns:
                        sector_mapper[asset.lower()] = class_name
                        print(f"  - {asset.lower()} mapped to sector {class_name}")
                    else:
                        print(
                            f"Warning: Asset {asset} not found in data, skipping from sector constraint"
                        )
                if "sum" in value:
                    sector_upper[class_name] = value["sum"]
                    print(f"  - Upper bound for sector {class_name}: {value['sum']}")

        # Check if the sum of max weights in weight_bounds is less than 1
        total_max_weight = sum(max_weight for _, max_weight in weight_bounds)
        if total_max_weight < 1:
            remaining_weight = 1 - total_max_weight
            if "usdc" in self.df.columns:
                usdc_index = self.df.columns.get_loc("usdc")
                current_usdc_max_weight = weight_bounds[usdc_index][1]
                new_usdc_max_weight = current_usdc_max_weight + remaining_weight
                weight_bounds[usdc_index] = (0, 1.0)
                print(f"Adjusting USDC max weight to: (0, {new_usdc_max_weight})")
            elif "btc" in self.df.columns:
                btc_index = self.df.columns.get_loc("btc")
                current_btc_max_weight = weight_bounds[btc_index][1]
                new_btc_max_weight = current_btc_max_weight + remaining_weight
                weight_bounds[btc_index] = (0, 1.0)
                print(f"Adjusting BTC max weight to: (0, {new_btc_max_weight})")
            elif "bnb" in self.df.columns:
                bnb_index = self.df.columns.get_loc("bnb")
                current_bnb_max_weight = weight_bounds[bnb_index][1]
                new_bnb_max_weight = current_bnb_max_weight + remaining_weight
                weight_bounds[bnb_index] = (0, 1.0)
                print(f"Adjusting BNB max weight to: (0, {new_bnb_max_weight})")

        print(f"Weight bounds: {weight_bounds}")
        print(f"Cov matrix: {self.cov_matrix}")
        print(f"Rets: {self.rets}")
        self.ef = EfficientFrontier(
            self.rets,
            self.cov_matrix,
            weight_bounds=weight_bounds,
            solver="ECOS_BB",
        )

        if sector_mapper and sector_upper:
            print("Applying sector constraints")
            self.ef.add_sector_constraints(
                sector_mapper, sector_lower={}, sector_upper=sector_upper
            )

        print("Constraints applied successfully.")
        print(self.ef._constraints)

    def get_weights(self, risk_free_rate=None):
        """
        Get the optimized portfolio weights.

        Parameters:
        -----------
        risk_free_rate : float, optional
            The risk-free rate to use in the optimization.

        Returns:
        --------
        weights : pandas.Series
            A pandas Series object containing the optimized weights for each asset in the portfolio.
        """

        self.delegate.setup(self)

        if self.df.shape[0] == 0 or self.df.shape[1] == 0:
            return pd.Series()

        if self.cov_matrix is None or self.cov_matrix.shape[0] == 0:
            self.cov_matrix = self.get_cov_matrix()
        if self.rets is None:
            self.rets = expected_returns.mean_historical_return(
                self.df, log_returns=True
            )

        self.process_constraints()

        risk_free_rate = (
            min(self.ef.expected_returns) if risk_free_rate is None else risk_free_rate
        )
        assert max(self.ef.expected_returns) >= risk_free_rate, (
            "Expected returns are less than the risk free rate. "
            "This is not possible. Please check your data."
        )

        try:
            if isinstance(self.efficient_portfolio, self.EfficientPortfolio):
                if self.efficient_portfolio == self.EfficientPortfolio.MAX_SHARPE:
                    self.ef.max_sharpe(risk_free_rate=risk_free_rate)
                elif self.efficient_portfolio == self.EfficientPortfolio.MIN_VOLATILITY:
                    self.ef.min_volatility()
                elif self.efficient_portfolio == self.EfficientPortfolio.MAX_RETURN:
                    self.ef.max_quadratic_utility()
            elif isinstance(self.efficient_portfolio, float):
                self.ef.efficient_return(self.efficient_portfolio)

            weights = pd.Series(self.ef.clean_weights())
            return weights
        except ValueError as e:
            print(f"Rets: {self.rets}")
            self.weights = pd.Series()
            return pd.Series()

    def check_constraints(self, weights):
        print("\nChecking constraints:")
        default_max = self.max_weights.get("*", 1)
        for asset, weight in weights.items():
            if asset in self.max_weights:
                print(
                    f"Individual constraint for {asset}: weight ({weight:.4f}) <= {self.max_weights[asset]}"
                )
            else:
                print(
                    f"Default constraint for {asset}: weight ({weight:.4f}) <= {default_max}"
                )

        for key, value in self.max_weights.items():
            if isinstance(value, dict) and "sum" in value:
                class_sum = sum(
                    weights.get(asset.lower(), 0) for asset in value.get("assets", [])
                )
                print(
                    f"Sector constraint for {key}: sum of weights ({class_sum:.4f}) <= {value['sum']}"
                )

        print(f"\nFinal weights:")
        for asset, weight in weights.items():
            print(f"{asset}: {weight:.4f}")

    def get_metrics(self):
        if self.ef.weights is None:
            return None
        metrics = self.ef.portfolio_performance(verbose=False)
        return {
            "apy": metrics[0],
            "annual_volatility": metrics[1],
            "sharpe_ratio": metrics[2],
        }

    def plot_frontier(self):
        fig, ax = plt.subplots()
        ef = EfficientFrontier(
            self.rets,
            self.cov_matrix,
            weight_bounds=(0, 1),
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
        if self.df.shape[0] == 0 or self.df.shape[1] == 0:
            return pd.DataFrame(columns=self.df.columns)
        return risk_matrix(self.df, method=self.mode.value)
