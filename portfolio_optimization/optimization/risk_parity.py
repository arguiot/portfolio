from enum import Enum
from .GeneralOptimization import GeneralOptimization
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pypfopt.risk_models import CovarianceShrinkage, sample_cov
from .risk_parity_impl import *


class RiskParity(GeneralOptimization):
    class Mode(Enum):
        SAMPLE_COV = 1
        LEDOIT_WOLF = 2

    def __init__(
        self, df: pd.DataFrame, mcaps=None, cov=None, asset_weight_bounds={"*": (0, 1)}
    ):
        super().__init__(df, mcaps=mcaps)

        self.asset_weight_bounds = asset_weight_bounds
        self.mode = self.Mode.LEDOIT_WOLF

        if cov is None:
            self.cov_matrix = self.get_cov_matrix()
        else:
            self.cov_matrix = cov

        self.budget = {}

        # Constraints
        self.tau: float | None = None
        self.gamma: float = 0.9
        self.zeta: float = 1e-7
        self.funtol: float = 0.000001
        self.wtol: float = 0.000001
        self.maxiter: int = 500
        self.Cmat: NDArray[np.float64] | None = None
        self.cvec: NDArray[np.float64] | None = None
        self.lambda_var: float | None = None
        self.lambda_u: float | None = None
        self.latest_apy: pd.Series | None = None

    def get_weights(self):
        self.delegate.setup(self)
        num_assets = self.df.shape[1]
        # Calculate budget
        # Normalizing the budget_target values to sum to the number of assets.
        norm_factor = (
            1 / (sum(self.budget.values()) + num_assets - len(self.budget))
            if len(self.budget) > 0
            else 1
        )
        budget_target_norm = {k: v * norm_factor for k, v in self.budget.items()}

        # Calculate default equally distributed budget.
        budget = np.ones(self.cov_matrix.shape[0]) / self.cov_matrix.shape[0]

        # Modify the budget for the specific assets.
        for i in range(num_assets):
            if self.df.columns[i] in budget_target_norm:
                budget = budget.at[i].set(budget_target_norm[self.df.columns[i]])

        # Initialize Dmat and dvec based on the weight boundaries for each asset
        self.Dmat: NDArray[np.float64] = np.vstack(
            [-np.eye(num_assets), np.eye(num_assets)]
        )  # 2n x n matrix
        self.process_asset_weight_bounds()
        min_vec = pd.Series(
            [self.asset_weight_bounds[asset][0] for asset in self.df.columns],
            index=self.df.columns,
        )
        max_vec = pd.Series(
            [self.asset_weight_bounds[asset][1] for asset in self.df.columns],
            index=self.df.columns,
        )
        self.dvec: NDArray[np.float64] = np.concatenate(
            [
                -np.ones(num_assets) * np.array(min_vec),
                np.ones(num_assets) * np.array(max_vec),
            ]
        )  # 2n vector

        # Optimize the portfolio
        pf = RiskParityPortfolio(
            covariance=self.cov_matrix,
            budget=budget,
        )

        # Set the constraints
        pf.add_variance(self.lambda_var) if hasattr(
            self, "lambda_var"
        ) and self.lambda_var is not None else None

        if (
            hasattr(self, "lambda_u")
            and self.lambda_u is not None
            and hasattr(self, "latest_apy")
            and self.latest_apy is not None
        ):
            # First, we need to align the assets in the latest_apy with the assets in the dataframe
            # This is because the latest_apy is calculated using the dataframe, but the dataframe
            # may have dropped some assets due to missing data.
            mu = self.latest_apy.reindex(self.df.columns, fill_value=0).values
            pf.add_mean_return(self.lambda_u, mu)

        pf.design(
            tau=self.tau,
            gamma=self.gamma,
            zeta=self.zeta,
            funtol=self.funtol,
            wtol=self.wtol,
            maxiter=self.maxiter,
            Cmat=self.Cmat,
            cvec=self.cvec,
            Dmat=self.Dmat,
            dvec=self.dvec,
        )
        weights = pd.Series(pf.weights, index=self.df.columns)
        return weights

    def get_metrics(self):
        pass

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
