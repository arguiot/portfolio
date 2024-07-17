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
    Markowitz portfolio optimization with multi-sector support.
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

    def __init__(self, df, mcaps=None, cov=None, max_weights=None, min_weights=None):
        super().__init__(df, mcaps=mcaps)

        self.mode = self.CovMode.LEDOIT_WOLF
        self.efficient_portfolio = self.EfficientPortfolio.MAX_SHARPE

        self.cov_matrix = cov
        self.rets = None
        self.max_weights = max_weights or {}
        self.min_weights = min_weights or {}
        self.ef = None
        self.sector_constraints = {}
        self.asset_constraints = {}
        self.yield_data = None

    def process_constraints(self):
        print("\nProcessing constraints:")
        self.sector_constraints = {}
        self.asset_constraints = {}
        default_max = self.max_weights.get("*", 1.0)

        available_assets = set(self.df.columns)

        # First pass: collect all constraints and identify missing assets
        for key, value in self.max_weights.items():
            if isinstance(value, (int, float)):
                if key == "*":
                    continue
                if key.lower() in available_assets:
                    self.asset_constraints[key.lower()] = (0, value)
                else:
                    print(
                        f"Warning: Asset {key} not found in data, skipping constraint"
                    )
            elif isinstance(value, dict):
                sector_name = key
                sector_allocation = value.get("sum", 0)
                sector_assets = [
                    asset.lower()
                    for asset in value.get("assets", [])
                    if asset.lower() in available_assets
                ]
                if sector_assets:
                    self.sector_constraints[sector_name] = {
                        "allocation": sector_allocation,
                        "assets": sector_assets,
                        "original_assets": value.get("assets", []),
                    }
                else:
                    print(
                        f"Warning: No assets found for sector {sector_name}, skipping"
                    )

        # Apply default constraints to unconstrained assets
        for asset in available_assets:
            if asset not in self.asset_constraints:
                self.asset_constraints[asset] = (0, default_max)

        # If no sector constraints, adjust if assets are missing
        if not self.sector_constraints or self.sector_constraints == {}:
            sum_weights = sum(
                self.asset_constraints[asset][1] for asset in available_assets
            )
            if sum_weights < 1:
                adjustment_factor = 1 / sum_weights
                for asset in available_assets:
                    original_max = self.max_weights.get(asset, default_max)
                    adjusted_max = max(
                        min(original_max * adjustment_factor, 1.0),
                        1 / (self.df.shape[1] - 1),
                    )
                    self.asset_constraints[asset] = (0, adjusted_max)
                    print(f"Adjusted constraint for {asset}: (0, {adjusted_max})")

        # Adjust max weights within each sector if assets are missing
        for sector, details in self.sector_constraints.items():
            sector_assets = details["assets"]
            original_assets_count = len(details["original_assets"])
            available_assets_count = len(sector_assets)
            if available_assets_count < original_assets_count:
                adjustment_factor = original_assets_count / available_assets_count
                for asset in sector_assets:
                    original_max = self.max_weights.get(asset, default_max)
                    adjusted_max = max(
                        min(original_max * adjustment_factor, 1.0),
                        1 / (available_assets_count),
                    )
                    self.asset_constraints[asset] = (0, adjusted_max)
                    print(
                        f"Adjusted constraint for {asset} in {sector}: (0, {adjusted_max})"
                    )
            else:
                for asset in sector_assets:
                    original_max = self.max_weights.get(asset, default_max)
                    sector_allocation = details["allocation"]
                    adjusted_max = max(
                        min(original_max / sector_allocation, 1.0),
                        1 / (available_assets_count),
                    )
                    self.asset_constraints[asset] = (0, adjusted_max)
                    print(
                        f"Adjusted constraint for {asset} in {sector}: (0, {adjusted_max})"
                    )

        # Ensure the sum of max weights within each sector is feasible
        for sector, details in self.sector_constraints.items():
            sector_assets = details["assets"]
            total_max_weight = sum(
                self.asset_constraints[asset][1] for asset in sector_assets
            )
            if total_max_weight < details["allocation"]:
                adjustment_factor = details["allocation"] / total_max_weight
                for asset in sector_assets:
                    min_weight, max_weight = self.asset_constraints[asset]
                    adjusted_max = min(max_weight * adjustment_factor, 1.0)
                    self.asset_constraints[asset] = (min_weight, adjusted_max)
                    print(
                        f"Re-adjusted constraint for {asset} in {sector}: (0, {adjusted_max})"
                    )

        print("\nFinal constraints:")
        for asset, (min_weight, max_weight) in self.asset_constraints.items():
            print(f"  {asset}: ({min_weight}, {max_weight})")
        for sector, details in self.sector_constraints.items():
            print(
                f"  Sector {sector}: allocation = {details['allocation']}, assets = {details['assets']}"
            )

    def merge_sector_portfolios(self, sector_portfolios):
        final_weights = {}
        for sector, details in self.sector_constraints.items():
            sector_allocation = details["allocation"]
            sector_weights = sector_portfolios[sector]
            for asset, weight in sector_weights.items():
                final_weights[asset] = weight * sector_allocation

        # Handle non-sector assets
        non_sector_assets = set(self.df.columns) - set(
            asset
            for details in self.sector_constraints.values()
            for asset in details["assets"]
        )
        if non_sector_assets:
            non_sector_allocation = 1 - sum(
                details["allocation"] for details in self.sector_constraints.values()
            )
            non_sector_weights = sector_portfolios.get("non_sector", {})
            for asset, weight in non_sector_weights.items():
                final_weights[asset] = weight * non_sector_allocation

        return final_weights

    def optimize_sector_portfolio(self, sector_assets, sector_allocation):
        sector_rets = self.rets[sector_assets]
        sector_cov = self.cov_matrix.loc[sector_assets, sector_assets]

        if len(sector_assets) == 1:
            # Return 100% weight if only one asset in sector
            return pd.Series(sector_allocation, index=sector_assets)

        weight_bounds = []
        for asset in sector_assets:
            min_weight, max_weight = self.asset_constraints[asset]
            scaled_max = max_weight
            weight_bounds.append((0, scaled_max))

        print(f"Sector: {sector_assets}")
        print(f"Sector: allocation = {sector_allocation}")
        print(f"Weight bounds: {weight_bounds}")

        ef = EfficientFrontier(
            sector_rets,
            sector_cov,
            weight_bounds=weight_bounds,
            solver="ECOS_BB",
        )

        # Set risk-free rate to the lowest expected return in the sector
        risk_free_rate = min(sector_rets) - 1e-4

        try:
            if self.efficient_portfolio == self.EfficientPortfolio.MAX_SHARPE:
                ef.max_sharpe(risk_free_rate=risk_free_rate)
            elif self.efficient_portfolio == self.EfficientPortfolio.MIN_VOLATILITY:
                ef.min_volatility()
            elif self.efficient_portfolio == self.EfficientPortfolio.MAX_RETURN:
                ef.max_quadratic_utility()

            weights = ef.clean_weights()
            return weights
        except Exception as e:
            print(f"Error optimizing sector portfolio: {str(e)}")
            raise  # Re-raise the exception to handle it in the calling function

    def get_weights(self, risk_free_rate=None):
        self.delegate.setup(self)

        if self.df.empty:
            return pd.Series()

        if self.cov_matrix is None or self.cov_matrix.empty:
            self.cov_matrix = self.get_cov_matrix()
        if self.rets is None:
            self.rets = expected_returns.mean_historical_return(
                self.df, log_returns=True
            )
        if self.yield_data is not None:
            for asset in self.yield_data.index:
                if asset not in self.rets.index:
                    continue
                # Convert annual yield to daily yield
                daily_yield = (1 + self.yield_data[asset]) ** (1 / 365) - 1

                # Convert log return to simple return
                simple_return = np.exp(self.rets[asset]) - 1

                # Add yield to simple return
                total_return = (1 + simple_return) * (1 + daily_yield) - 1

                # Convert back to log return
                self.rets[asset] = np.log(1 + total_return)

        self.process_constraints()

        if self.sector_constraints:
            # Sector-based portfolio
            sector_portfolios = {}
            for sector, details in self.sector_constraints.items():
                print(f"\nProcessing sector: {sector}")
                try:
                    sector_weights = self.optimize_sector_portfolio(
                        details["assets"], details["allocation"]
                    )
                    sector_portfolios[sector] = sector_weights
                except Exception as e:
                    print(f"Failed to optimize sector {sector}: {str(e)}")
                    return pd.Series()

            final_weights = self.merge_sector_portfolios(sector_portfolios)
        else:
            # Non-sector-based portfolio
            all_assets = list(self.df.columns)

            final_weights = self.optimize_sector_portfolio(all_assets, 1)

        # Verify individual asset constraints
        for asset, weight in final_weights.items():
            if weight > self.asset_constraints[asset][1] + 1e-4:
                print(
                    f"Error: Asset {asset} exceeds maximum weight. Expected <= {self.asset_constraints[asset][1]}, got {weight}"
                )
                return pd.Series()

        # Verify total allocation
        total_weight = sum(final_weights.values())
        if abs(total_weight - 1) > 1e-4:
            # Adjust weights if total weight is not 1
            for asset, weight in final_weights.items():
                final_weights[asset] = weight / total_weight

        return pd.Series(final_weights)

    def check_constraints(self, weights):
        print("\nChecking constraints:")
        for sector, details in self.sector_constraints.items():
            sector_sum = sum(weights.get(asset, 0) for asset in details["assets"])
            print(
                f"Sector {sector}: sum of weights ({sector_sum:.4f}) == {details['allocation']}"
            )

        for asset, weight in weights.items():
            constraint = self.asset_constraints.get(asset, (0, 1))
            print(f"Asset {asset}: weight ({weight:.4f}) in range {constraint}")

        print("\nFinal weights:")
        for asset, weight in weights.items():
            print(f"{asset}: {weight:.4f}")

    def get_metrics(self):
        if not hasattr(self, "weights") or self.weights is None:
            return None

        portfolio_return = np.sum(self.rets * self.weights)
        portfolio_volatility = np.sqrt(
            np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights))
        )
        sharpe_ratio = portfolio_return / portfolio_volatility

        return {
            "apy": portfolio_return,
            "annual_volatility": portfolio_volatility,
            "sharpe_ratio": sharpe_ratio,
        }

    def plot_frontier(self):
        # This method might need to be updated to handle multi-sector portfolios
        # For now, we'll keep the original implementation
        fig, ax = plt.subplots()
        ef = EfficientFrontier(
            self.rets,
            self.cov_matrix,
            weight_bounds=(0, 1),
            solver="ECOS_BB",
        )
        ef_max_sharpe = ef.deepcopy()
        plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

        ef_max_sharpe.max_sharpe()
        ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
        ax.scatter(
            std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe"
        )

        n_samples = 15000
        w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
        rets = w.dot(ef.expected_returns)
        stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
        sharpes = rets / stds
        ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

        ax.set_title("Efficient Frontier with random portfolios")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def get_cov_matrix(self):
        if self.df.empty:
            return pd.DataFrame(columns=self.df.columns)
        return risk_matrix(self.df, method=self.mode.value)
