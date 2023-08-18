from .GeneralOptimization import GeneralOptimization
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class RiskParity(GeneralOptimization):
    def __init__(self, df, mcaps=None, weight_bounds=(0, 1)):
        self.weight_bounds = weight_bounds
        super().__init__(df, mcaps)

    # risk budgeting optimization
    def calculate_portfolio_var(self, w, V):
        # function that calculates portfolio risk
        w = np.matrix(w)
        return (w * V * w.T)[0, 0]

    def calculate_risk_contribution(self, w, V):
        # function that calculates asset contribution to total risk
        w = np.matrix(w)
        sigma = np.sqrt(self.calculate_portfolio_var(w, V))
        # Marginal Risk Contribution
        MRC = V * w.T
        # Risk Contribution
        RC = np.multiply(MRC, w.T) / sigma
        return RC

    def risk_budget_objective(self, x, pars):
        # calculate portfolio risk
        V = pars[0]  # covariance table
        x_t = pars[1]  # risk target in percent of portfolio risk
        sig_p = np.sqrt(self.calculate_portfolio_var(x, V))  # portfolio sigma
        risk_target = np.asmatrix(np.multiply(sig_p, x_t))
        asset_RC = self.calculate_risk_contribution(x, V)
        J = sum(np.square(asset_RC - risk_target.T))[0, 0]  # sum of squared error
        return J

    def total_weight_constraint(self, x):
        return np.sum(x) - 1.0

    def long_only_constraint(self, x):
        return self.weight_bounds[1] - x

    def short_only_constraint(self, x):
        return x - self.weight_bounds[0]  # Ensuring weights are not less than

    def get_weights(self):
        # 1 / N risk portfolio
        x_t = [1 / self.df.shape[1]] * self.df.shape[1]
        cons = (
            {"type": "eq", "fun": self.total_weight_constraint},
            {"type": "ineq", "fun": self.short_only_constraint},
            {"type": "ineq", "fun": self.long_only_constraint},
        )

        w0 = np.ones(self.df.shape[1]) * (1.0 / self.df.shape[1],)
        V = np.cov(self.df.T)

        res = minimize(
            self.risk_budget_objective,
            w0,
            args=[V, x_t],
            method="SLSQP",
            constraints=cons,
            options={"disp": True, "ftol": 1e-9},
        )
        weights = pd.Series(res.x, index=self.df.columns)
        return weights

    def get_metrics(self):
        pass
