from .GeneralOptimization import GeneralOptimization
from pypfopt.hierarchical_portfolio import HRPOpt
from ..data_processing.expected_returns import expected_returns
import pandas as pd
import riskfolio as rp


class HRPOptimization(GeneralOptimization):
    """
    A class used to perform hierarchical risk parity portfolio optimization.

    Inherits from the GeneralOptimization class and uses the HRPOpt algorithm from the pypfopt library.

    ...

    Attributes
    ----------
    df : pandas.DataFrame
        a pandas DataFrame containing historical asset prices
    rets : pandas.DataFrame, optional
        a pandas DataFrame containing historical asset returns (default is None)

    Methods
    -------
    optimize()
        Optimize the portfolio weights using the HRPOpt algorithm.
    clean_weights()
        Clean the optimized weights to remove any assets with zero weight.
    get_weights()
        Optimize the portfolio weights and return the cleaned weights.

    """

    def __init__(self, df, mcaps=None, rets=None, asset_weight_bounds={"*": (0, 1)}):
        super().__init__(df, mcaps=mcaps)
        self.asset_weight_bounds = asset_weight_bounds
        if rets is None or rets.empty:
            self.rets = expected_returns(df)
        else:
            self.rets = rets

        self.rets = self.rets.fillna(0)
        self.port = rp.HCPortfolio(returns=self.rets)

    def optimize(self):
        """
        Optimize the portfolio weights using the HRPOpt algorithm.

        This method uses the HRPOpt algorithm from the pypfopt library to optimize the portfolio weights.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        if self.df is None or self.df.shape[1] == 0:
            self.weights = pd.Series()
            return
        if self.rets is None or self.rets.shape[0] == 0:
            self.rets = expected_returns(self.df)
            # print(f"Rets: {self.rets}")
            # print(f"df: {self.df}")

        self.process_asset_weight_bounds()
        min_vec = pd.Series(
            [self.asset_weight_bounds[asset][0] for asset in self.df.columns],
            index=self.df.columns,
        )
        max_vec = pd.Series(
            [self.asset_weight_bounds[asset][1] for asset in self.df.columns],
            index=self.df.columns,
        )

        total_upper_bound = max_vec.sum()
        if total_upper_bound < 1:
            if "usdc" in self.df.columns:
                max_vec["usdc"] = 1.0
            elif "btc" in self.df.columns:
                max_vec["btc"] = 1.0
            elif "bnb" in self.df.columns:
                max_vec["bnb"] = 1.0

        self.port.w_min = min_vec
        self.port.w_max = max_vec

        print(self.df)

        self.weights = self.port.optimization(
            model="HRP",
            codependence="pearson",
            rm="MV",
            rf=0.0,
            linkage="single",
            leaf_order=True,
        )["weights"]

    def clean_weights(self):
        """
        Clean the optimized weights to remove any assets with zero weight.

        This method removes any assets with zero weight from the optimized weights.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # self.weights = self.weights.loc[self.weights > 1e-4]

    def get_weights(self):
        """
        Optimize the portfolio weights and return the cleaned weights.

        This method optimizes the portfolio weights using the HRPOpt algorithm from the pypfopt library and
        returns the cleaned weights with any assets with zero weight removed.

        Parameters
        ----------
        None

        Returns
        -------
        pandas.DataFrame
            a pandas DataFrame containing the optimized and cleaned portfolio weights

        """
        self.delegate.setup(self)
        self.optimize()
        # self.clean_weights()
        return self.weights

    def get_metrics(self):
        pass
