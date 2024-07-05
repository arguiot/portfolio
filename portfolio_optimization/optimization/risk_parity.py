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

    def __init__(self, df: pd.DataFrame, mcaps=None, cov=None, max_weight=None):
        super().__init__(df, mcaps=mcaps)

        self.mode = self.Mode.LEDOIT_WOLF
        self.max_weights = max_weight or {}

        self.asset_names = list(df.columns)
        if cov is None:
            self.cov_matrix = self.get_cov_matrix()
        else:
            self.cov_matrix = cov

        self.budget = {}
        self.returns = None

        # Constraints
        self.lambda_var = None
        self.lambda_u = None
        self.latest_apy = None

        # Preprocess assets and constraints
        self.valid_assets = self._get_valid_assets()
        self.processed_max_weights = self._process_max_weights()

    def _get_valid_assets(self):
        return set(self.asset_names)

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

    def get_weights(self):
        self.delegate.setup(self)
        if not self.validate_constraints():
            raise ValueError(
                "Constraints are not mathematically consistent. Please review your max_weight input."
            )
        if self.returns is None:
            self.returns = expected_returns.mean_historical_return(
                self.df, log_returns=True
            )

        self.cov_matrix = self.get_cov_matrix()

        N = len(self.valid_assets)
        w = cp.Variable((N, 1))
        rb = np.ones((N, 1)) / N

        k = cp.Variable((1, 1))

        # MV Model Variables
        risk = cp.quad_form(w, cp.psd_wrap(self.cov_matrix))

        constraints = (
            []
        )  # [cp.SOC(1, G @ w)]  # This replaces the previous SOC constraint

        # Risk budgeting constraint
        log_w = cp.Variable((N, 1))
        constraints += [
            w * 1000 >= 0,
            rb.T @ log_w >= 1,
            cp.ExpCone(log_w * 1000, np.ones((N, 1)) * 1000, w * 1000),
        ]

        constraints += [
            cp.sum(w) * 1000 == k * 1000,
        ]

        # Add max weight constraints
        total_max_weight = 0
        for i, asset in enumerate(self.valid_assets):
            max_weight = self.processed_max_weights.get(
                asset.lower(), self.processed_max_weights.get("*", 1.0)
            )
            if type(max_weight) == float:
                constraints.append(w[i] <= max_weight * k)
                total_max_weight += max_weight

        # Check if the sum of max weights is less than 2
        if total_max_weight < 1:
            remaining_weight = 1 - total_max_weight
            if "usdc" in self.valid_assets:
                usdc_index = list(self.valid_assets).index("usdc")
                current_usdc_max_weight = self.processed_max_weights.get("usdc", 0)
                new_usdc_max_weight = current_usdc_max_weight + remaining_weight
                constraints[usdc_index] = w[usdc_index] <= 1.0 * k
                print(f"Adjusting USDC max weight to: {new_usdc_max_weight}")
            elif "btc" in self.valid_assets:
                btc_index = list(self.valid_assets).index("btc")
                current_btc_max_weight = self.processed_max_weights.get("btc", 0)
                new_btc_max_weight = current_btc_max_weight + remaining_weight
                constraints[btc_index] = w[btc_index] <= 1.0 * k
                print(f"Adjusting BTC max weight to: {new_btc_max_weight}")
            elif "bnb" in self.valid_assets:
                bnb_index = list(self.valid_assets).index("bnb")
                current_bnb_max_weight = self.processed_max_weights.get("bnb", 0)
                new_bnb_max_weight = current_bnb_max_weight + remaining_weight
                constraints[bnb_index] = w[bnb_index] <= 1.0 * k
                print(f"Adjusting BNB max weight to: {new_bnb_max_weight}")

        # Add class constraints
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
                    print(f"Adding class constraint for {key}: sum <= {value['sum']}")
                    constraints.append(class_selector @ w <= value["sum"] * k)

        # Objective function
        objective = cp.Minimize(risk * 1000)

        # Solve the problem
        prob = cp.Problem(objective, constraints)

        for solver in ["MOSEK", "CLARABEL", "SCS", "ECOS_BB"]:
            try:
                prob.solve(solver=solver, verbose=True)

                if w.value is not None:
                    break
            except cp.error.SolverError:
                print(f"Solver {solver} failed. Trying next solver.")

        if w.value is None:
            print("Optimization status:", prob.status)
            print("Optimal value:", prob.value)
            for i, constraint in enumerate(constraints):
                try:
                    print(f"Constraint {i} violation:", constraint.violation())
                except ValueError:
                    print(f"Constraint {i} could not be evaluated")
            fallback = FastRiskParity(self.df, mcaps=self.mcaps)
            return fallback.get_weights()
            # raise ValueError(
            #     f"Optimization failed to find a solution. Current df: {self.df}. Total max weight: {total_max_weight}. Cov matrix: {self.cov_matrix}"
            # )

        weights = np.abs(w.value) / np.sum(np.abs(w.value))
        return pd.Series(weights.flatten(), index=self.valid_assets)

    def validate_constraints(self) -> bool:
        # Preprocess assets and constraints
        self.valid_assets = self._get_valid_assets()
        self.processed_max_weights = self._process_max_weights()

        assets_in_constraints = set()

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
                assets_in_constraints.update(
                    [asset.lower() for asset in value["assets"]]
                )

        for key, value in self.processed_max_weights.items():
            if isinstance(value, dict):
                class_assets = set([asset.lower() for asset in value["assets"]])
                for other_key, other_value in self.processed_max_weights.items():
                    if other_key != key and isinstance(other_value, dict):
                        other_class_assets = set(
                            [asset.lower() for asset in other_value["assets"]]
                        )
                        if class_assets.intersection(other_class_assets):
                            print(
                                f"Warning: Overlapping assets in class constraints {key} and {other_key}"
                            )

        unconstrained_assets = self.valid_assets - assets_in_constraints
        if unconstrained_assets and "*" not in self.processed_max_weights:
            print(
                f"Warning: The following assets have no explicit constraints: {unconstrained_assets}"
            )

        return True

    def get_metrics(self):
        pass

    def get_cov_matrix(self):
        if self.mode == self.Mode.SAMPLE_COV:
            cov = np.array(sample_cov(self.df))
        elif self.mode == self.Mode.LEDOIT_WOLF:
            cov = np.array(CovarianceShrinkage(self.df).ledoit_wolf())
        print(f"Computed covariance matrix shape: {cov.shape}")
        return cov

    def check_constraints(self, weights):
        print("\nChecking constraints:")
        default_max = self.processed_max_weights.get("*", 1.0)
        for asset, weight in weights.items():
            if asset in self.processed_max_weights:
                max_weight = min(self.processed_max_weights[asset], default_max)
                print(
                    f"Individual constraint for {asset}: weight ({weight:.4f}) <= {max_weight}"
                )
            else:
                print(
                    f"Default constraint for {asset}: weight ({weight:.4f}) <= {default_max}"
                )

        for key, value in self.processed_max_weights.items():
            if isinstance(value, dict) and "sum" in value:
                class_sum = sum(
                    weights.get(asset.lower(), 0) for asset in value.get("assets", [])
                )
                max_sum = min(value["sum"], default_max)
                print(
                    f"Sector constraint for {key}: sum of weights ({class_sum:.4f}) <= {max_sum}"
                )

        print(f"\nFinal weights:")
        for asset, weight in weights.items():
            print(f"{asset}: {weight:.4f}")
