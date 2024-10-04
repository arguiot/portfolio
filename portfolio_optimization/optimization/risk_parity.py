from enum import Enum
from .GeneralOptimization import GeneralOptimization
import cvxpy as cp
import numpy as np
import pandas as pd
from pypfopt.risk_models import CovarianceShrinkage, sample_cov
from pypfopt import expected_returns
from portfolio_optimization.optimization.heuristic import FastRiskParity


class RiskParity(GeneralOptimization):
    class Mode(Enum):
        SAMPLE_COV = 1
        LEDOIT_WOLF = 2

    def __init__(
        self, df: pd.DataFrame, mcaps=None, cov=None, max_weight=None, min_weight=None
    ):
        super().__init__(df, mcaps=mcaps)
        self.mode = self.Mode.LEDOIT_WOLF
        self.max_weights = max_weight or {}
        self.min_weights = min_weight or {}
        self.asset_names = sorted(list(df.columns))  # Sort asset names
        self.returns = None
        self.df = df[self.asset_names]  # Reorder DataFrame columns
        self.cov_matrix = cov if cov is not None else self.get_cov_matrix()
        self.budget = {}
        self.lambda_var = None
        self.lambda_u = None
        self.latest_apy = None
        self.valid_assets = self._get_valid_assets()
        self.processed_max_weights = self._process_max_weights()
        self.sector_constraints = {}
        self.asset_constraints = {}

    def _get_valid_assets(self):
        return sorted(list(self.asset_names))

    def _process_max_weights(self):
        processed = {}
        for key, value in self.max_weights.items():
            if isinstance(value, (int, float)):
                if key == "*" or key.lower() in self.valid_assets:
                    processed[key] = value
            elif isinstance(value, dict) and "sum" in value:
                valid_assets = [
                    asset
                    for asset in value.get("assets", [])
                    if asset.lower() in self.valid_assets
                ]
                if valid_assets:
                    processed[key] = {"sum": value["sum"], "assets": valid_assets}
        return processed

    def process_constraints(self):
        # print("\nProcessing constraints:")
        self.sector_constraints = {}
        self.asset_constraints = {}
        default_max = self.processed_max_weights.get("*", 1.0)

        available_assets = list(self.df.columns)

        for key, value in self.processed_max_weights.items():
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
                    # print(f"Adjusted constraint for {asset}: (0, {adjusted_max})")

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
                    # print(
                    #     f"Adjusted constraint for {asset} in {sector}: (0, {adjusted_max})"
                    # )
            else:
                for asset in sector_assets:
                    original_max = self.max_weights.get(asset, default_max)
                    sector_allocation = details["allocation"]
                    adjusted_max = max(
                        min(original_max / sector_allocation, 1.0),
                        1 / (available_assets_count),
                    )
                    self.asset_constraints[asset] = (0, adjusted_max)
                    # print(
                    #     f"Adjusted constraint for {asset} in {sector}: (0, {adjusted_max})"
                    # )

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
                    # print(
                    #     f"Re-adjusted constraint for {asset} in {sector}: (0, {adjusted_max})"
                    # )

        # print("\nFinal constraints:")
        # for asset, (min_weight, max_weight) in self.asset_constraints.items():
        #     print(f"  {asset}: ({min_weight}, {max_weight})")
        # for sector, details in self.sector_constraints.items():
        #     print(
        #         f"  Sector {sector}: allocation = {details['allocation']}, assets = {details['assets']}"
        #     )

    def optimize_sector_portfolio(self, sector_assets, sector_allocation):
        sector_df = self.df[sector_assets]
        sector_indices = [self.asset_names.index(asset) for asset in sector_assets]
        sector_cov = self.cov_matrix[np.ix_(sector_indices, sector_indices)]

        N = len(sector_assets)
        w = cp.Variable((N, 1))
        rb = np.ones((N, 1)) / N
        k = cp.Variable((1, 1))

        risk = cp.quad_form(w, cp.psd_wrap(sector_cov))
        constraints = [
            w * 1000 >= 0,
            cp.sum(w) * 1000 == k * 1000,
        ]

        log_w = cp.Variable((N, 1))
        constraints += [
            rb.T @ log_w >= 1,
            cp.ExpCone(log_w * 1000, np.ones((N, 1)) * 1000, w * 1000),
        ]

        for i, asset in enumerate(sector_assets):
            max_weight = self.asset_constraints[asset][1]
            constraints.append(w[i] <= max_weight * k)

        objective = cp.Minimize(risk * 1000)
        prob = cp.Problem(objective, constraints)

        for solver in ["SCS", "ECOS_BB", "MOSEK", "CLARABEL"]:
            try:
                prob.solve(solver=solver, verbose=False)
                if w.value is not None:
                    break
            except cp.error.SolverError:
                print(f"Solver {solver} failed. Trying next solver.")

        if w.value is None:
            raise ValueError(f"Optimization failed for sector {sector_assets}")

        weights = np.abs(w.value) / np.sum(np.abs(w.value))
        return pd.Series(weights.flatten(), index=sector_assets)

    def merge_sector_portfolios(self, sector_portfolios):
        final_weights = {}
        for sector, details in self.sector_constraints.items():
            sector_allocation = details["allocation"]
            sector_weights = sector_portfolios[sector]
            for asset, weight in sector_weights.items():
                final_weights[asset] = weight * sector_allocation

        sector_assets = [
            asset
            for details in self.sector_constraints.values()
            for asset in details["assets"]
        ]
        non_sector_assets = [
            asset for asset in self.df.columns if asset not in sector_assets
        ]
        if non_sector_assets:
            non_sector_allocation = 1 - sum(
                details["allocation"] for details in self.sector_constraints.values()
            )
            non_sector_weights = sector_portfolios.get("non_sector", {})
            for asset, weight in non_sector_weights.items():
                final_weights[asset] = weight * non_sector_allocation

        return final_weights

    def get_weights(self):
        self.delegate.setup(self)
        if not self.validate_constraints():
            raise ValueError(
                "Constraints are not mathematically consistent. Please review your max_weight input."
            )

        self.cov_matrix = self.get_cov_matrix()
        self.process_constraints()

        if self.sector_constraints:
            sector_portfolios = {}
            for sector, details in self.sector_constraints.items():
                # print(f"\nProcessing sector: {sector}")
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
            try:
                final_weights = self.optimize_sector_portfolio(self.valid_assets, 1)
            except Exception as e:
                print(f"Failed to optimize portfolio: {str(e)}")
                return pd.Series()

        # print(f"Final weights: {final_weights}")

        return pd.Series(final_weights).sort_index()

    def _optimize_risk_parity(self):
        N = len(self.valid_assets)
        sorted_assets = sorted(self.valid_assets)  # Sort assets
        w = cp.Variable((N, 1))
        rb = np.ones((N, 1)) / N
        k = cp.Variable((1, 1))

        risk = cp.quad_form(w, cp.psd_wrap(self.cov_matrix))
        constraints = [
            w * 1000 >= 0,
            cp.sum(w) * 1000 == k * 1000,
        ]

        log_w = cp.Variable((N, 1))
        constraints += [
            rb.T @ log_w >= 1,
            cp.ExpCone(log_w * 1000, np.ones((N, 1)) * 1000, w * 1000),
        ]

        total_max_weight = 0
        for i, asset in enumerate(self.valid_assets):
            max_weight = self.processed_max_weights.get(
                asset.lower(), self.processed_max_weights.get("*", 1.0)
            )
            if type(max_weight) == float:
                constraints.append(w[i] <= max_weight * k)
                total_max_weight += max_weight

        if total_max_weight < 1:
            remaining_weight = 1 - total_max_weight
            if "usdc" in self.valid_assets:
                usdc_index = list(self.valid_assets).index("usdc")
                constraints[usdc_index] = w[usdc_index] <= 1.0 * k
                # print(f"Adjusting USDC max weight to: {remaining_weight}")
            elif "btc" in self.valid_assets:
                btc_index = list(self.valid_assets).index("btc")
                constraints[btc_index] = w[btc_index] <= 1.0 * k
                # print(f"Adjusting BTC max weight to: {remaining_weight}")
            elif "bnb" in self.valid_assets:
                bnb_index = list(self.valid_assets).index("bnb")
                constraints[bnb_index] = w[bnb_index] <= 1.0 * k
                # print(f"Adjusting BNB max weight to: {remaining_weight}")

        for key, value in self.processed_max_weights.items():
            if isinstance(value, dict) and "sum" in value:
                class_indices = [
                    list(self.valid_assets).index(asset.lower())
                    for asset in value["assets"]
                    if asset.lower() in self.valid_assets
                ]
                if class_indices:
                    class_selector = np.zeros((1, N))
                    class_selector[0, class_indices] = 1
                    # print(f"Adding class constraint for {key}: sum <= {value['sum']}")
                    constraints.append(class_selector @ w <= value["sum"] * k)

        objective = cp.Minimize(risk * 1000)
        prob = cp.Problem(objective, constraints)

        for solver in ["SCS", "ECOS_BB", "MOSEK", "CLARABEL"]:
            try:
                prob.solve(solver=solver, verbose=False)
                if w.value is not None:
                    break
            except cp.error.SolverError:
                print(f"Solver {solver} failed. Trying next solver.")

        if w.value is None:
            # print("Optimization status:", prob.status)
            # print("Optimal value:", prob.value)
            for i, constraint in enumerate(constraints):
                try:
                    print(f"Constraint {i} violation:", constraint.violation())
                except ValueError:
                    print(f"Constraint {i} could not be evaluated")
            fallback = FastRiskParity(self.df, mcaps=self.mcaps)
            return fallback.get_weights()

        weights = np.abs(w.value) / np.sum(np.abs(w.value))
        return pd.Series(weights.flatten(), index=sorted_assets).sort_index()

    def validate_constraints(self):
        self.valid_assets = self._get_valid_assets()
        self.processed_max_weights = self._process_max_weights()

        assets_in_constraints = set()  # Change this to a set

        for key, value in self.processed_max_weights.items():
            if isinstance(value, (int, float)):
                if key == "*":
                    if value <= 0 or value > 1:
                        print(
                            f"Error: Invalid global constraint ({value}). Must be between 0 and 1."
                        )
                        return False
                else:
                    if value < 0 or value > 1:
                        print(
                            f"Error: Invalid constraint for {key} ({value}). Must be between 0 and 1."
                        )
                        return False
                    assets_in_constraints.add(key.lower())
            elif isinstance(value, dict):
                if value["sum"] < 0 or value["sum"] > 1:
                    print(
                        f"Error: Invalid sum constraint for {key} ({value['sum']}). Must be between 0 and 1."
                    )
                    return False
                assets_in_constraints.update(asset.lower() for asset in value["assets"])

        for key, value in self.processed_max_weights.items():
            if isinstance(value, dict):
                class_assets = set(asset.lower() for asset in value["assets"])
                for other_key, other_value in self.processed_max_weights.items():
                    if other_key != key and isinstance(other_value, dict):
                        other_class_assets = set(
                            asset.lower() for asset in other_value["assets"]
                        )
                        if class_assets.intersection(other_class_assets):
                            print(
                                f"Warning: Overlapping assets in class constraints {key} and {other_key}"
                            )

        unconstrained_assets = set(self.valid_assets) - assets_in_constraints
        if unconstrained_assets and "*" not in self.processed_max_weights:
            print(
                f"Warning: The following assets have no explicit constraints: {unconstrained_assets}"
            )

        return True

    def get_metrics(self):
        pass

    def get_cov_matrix(self):
        if self.mode == self.Mode.SAMPLE_COV:
            cov = np.array(sample_cov(self.df, frequency=365))
        elif self.mode == self.Mode.LEDOIT_WOLF:
            cov = np.array(CovarianceShrinkage(self.df, frequency=365).ledoit_wolf())
        print(f"Computed covariance matrix shape: {cov.shape}.")
        return cov

    def check_constraints(self, weights):
        # print("\nChecking constraints:")
        # for sector, details in self.sector_constraints.items():
        #     sector_sum = sum(weights.get(asset, 0) for asset in details["assets"])
        #     print(
        #         f"Sector {sector}: sum of weights ({sector_sum:.4f}) == {details['allocation']}"
        #     )

        # for asset, weight in weights.items():
        #     constraint = self.asset_constraints.get(asset, (0, 1))
        #     print(f"Asset {asset}: weight ({weight:.4f}) in range {constraint}")

        # print("\nFinal weights:")
        # for asset, weight in weights.items():
        #     print(f"{asset}: {weight:.4f}")
        return
