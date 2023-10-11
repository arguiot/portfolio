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

    def __init__(self, df, mcaps=None, cov=None, weight_bounds=(0, 1)):
        super().__init__(df, mcaps=mcaps)

        self.weight_bounds = weight_bounds
        self.mode = self.Mode.LEDOIT_WOLF

        if cov is None:
            self.cov_matrix = self.get_cov_matrix()
        else:
            self.cov_matrix = cov

        # Contrainsts
        self.tau: float | None = None
        self.gamma: float = 0.9
        self.zeta: float = 1e-7
        self.funtol: float = 0.000001
        self.wtol: float = 0.000001
        self.maxiter: int = 500
        self.Cmat: NDArray[np.float64] | None = None
        self.cvec: NDArray[np.float64] | None = None
        self.Dmat: NDArray[np.float64] | None = None
        self.dvec: NDArray[np.float64] | None = None

    def get_weights(self):
        # Calculate budget, which is a vector of equal weights
        budget = np.ones(self.cov_matrix.shape[0]) / self.cov_matrix.shape[0]

        # Optimize the portfolio
        pf = RiskParityPortfolio(
            covariance=self.cov_matrix,
            budget=budget,
        )
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
