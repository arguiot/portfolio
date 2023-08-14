from .GeneralOptimization import GeneralOptimization
from pypfopt.hierarchical_portfolio import HRPOpt
from ..data_processing.expected_returns import expected_returns
import pandas as pd


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

    def __init__(self, df, mcaps=None, rets=None):
        super().__init__(df, mcaps=mcaps)
        if rets is None:
            self.rets = expected_returns(df)
        else:
            self.rets = rets
        self.rets = self.rets.fillna(0)
        self.hrp = HRPOpt(self.rets)

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
        self.hrp.optimize()

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
        self.weights = pd.Series(self.hrp.clean_weights())

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
        self.optimize()
        self.clean_weights()
        return self.weights

    def get_metrics(self):
        """
        Get the metrics, such as performances, for the optimized portfolio.

        This method returns the metrics for the optimized portfolio.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            a dictionary containing the metrics for the optimized portfolio

        """
        if self.hrp.weights is None:
            return None
        metrics = self.hrp.portfolio_performance(verbose=False, frequency=365)
        return {
            "apy": metrics[0],
            "annual_volatility": metrics[1],
            "sharpe_ratio": metrics[2],
        }
