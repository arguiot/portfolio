from .GeneralOptimization import GeneralOptimization
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from scipy import optimize

import cvxopt as opt
from cvxopt import solvers


class VaultAllocation(GeneralOptimization):
    """
    Vault Allocation portfolio optimization proposed by Bastien Baude.
    """

    def __init__(self, df, mcaps=None, asset_weight_bounds={"*": (0, 1)}):
        super().__init__(df, mcaps=mcaps)

        self.rets = None
        self.asset_weight_bounds = asset_weight_bounds
        self.ef = None
        # Yield Data
        self.yield_data: pd.Series | None = None
        # Recovery Rate
        self.recovery_rate: pd.Series | None = None
        # Probability of Default
        self.default_risk: pd.Series | None = None
        # Risk
        self.risk = 0.6
        # Diversification Factor
        self.diversification = 0.7

    def get_weights(self, risk_free_rate=None):
        self.delegate.setup(self)

        if (
            (self.df.empty and self.yield_data is None)
            or self.recovery_rate is None
            or self.default_risk is None
        ):
            return pd.Series()

        mu = self.yield_data if self.yield_data is not None else self.df.mean()
        R = self.recovery_rate
        lmb = self.default_risk
        omega = self.risk
        D = self.diversification

        EL = (1 - R) * lmb

        mu_tilde: pd.Series = mu / mu.sum()

        EL_tilde: pd.Series = EL / EL.sum()

        self.process_asset_weight_bounds(columns=mu.index)
        min_vec = pd.Series(
            [self.asset_weight_bounds[asset][0] for asset in mu.index],
            index=mu.index,
        )
        max_vec = pd.Series(
            [self.asset_weight_bounds[asset][1] for asset in mu.index],
            index=mu.index,
        )

        final_weights = self.optimal_vault_allocation(
            len(mu),
            min_vec.to_numpy(),
            max_vec.to_numpy(),
            mu_tilde.to_numpy(),
            EL_tilde.to_numpy(),
            omega,
            D,
        )

        return pd.Series(final_weights, index=mu.index)

    # algorithm
    def optimal_vault_allocation(
        self,
        n: int,
        weights_min: NDArray,
        weights_max: NDArray,
        mu_tilde: NDArray,
        EL_tilde: NDArray,
        omega: float,
        D: float,
    ):
        solvers.options["show_progress"] = False
        # compute the optimal v from the target diversification factor D
        nu_opt = self.optimal_nu(
            D, n, weights_min, weights_max, mu_tilde, EL_tilde, omega
        )

        # compute the optimal vault allocation
        portfolio = self.optimal_vault_allocation_given_v(
            n, weights_min, weights_max, mu_tilde, EL_tilde, omega, nu_opt
        )

        return np.reshape(
            portfolio,
            [
                -1,
            ],
        )

    def optimal_vault_allocation_given_v(
        self, n, weights_min, weights_max, mu_tilde, EL_tilde, omega, nu
    ):
        P = opt.matrix(nu * np.eye(n))
        q = opt.matrix(omega * EL_tilde.T - (1 - omega) * mu_tilde)

        G = opt.matrix(np.block([np.eye(n), -np.eye(n)]).T)
        h = opt.matrix(np.block([weights_max, -weights_min]), tc="d")
        A = opt.matrix(1.0, (1, n))
        b = opt.matrix(1.0)

        return solvers.qp(P, q, G, h, A, b)["x"]

    def helper_nu(self, y, D, n, weights_min, weights_max, mu_tilde, EL_tilde, omega):
        current_portfolio = np.reshape(
            self.optimal_vault_allocation_given_v(
                n, weights_min, weights_max, mu_tilde, EL_tilde, omega, y
            ),
            [-1, 1],
        )
        return 1 / np.sum(current_portfolio**2) / n - D

    def optimal_nu(self, D, n, weights_min, weights_max, mu_tilde, EL_tilde, omega):
        nu_opt = optimize.brentq(
            self.helper_nu,
            0,
            100,
            args=(D, n, weights_min, weights_max, mu_tilde, EL_tilde, omega),
            xtol=1e-4,
            rtol=1e-4,
        )

        return nu_opt

    def get_metrics(self):
        portfolio = self.get_weights()

        if portfolio.empty:
            return None

        mu = self.yield_data if self.yield_data is not None else self.df.last("1D")
        R = self.recovery_rate
        lmb = self.default_risk

        EL = (1 - R) * lmb
        n = len(mu)

        return {
            "Expected loss (%)": np.round(portfolio @ EL * 100, 2),
            "Yield (%)": np.round(portfolio @ mu * 100, 2),
            "Diversification (%)": np.round(1 / np.sum(portfolio**2) / n * 100, 2),
        }
