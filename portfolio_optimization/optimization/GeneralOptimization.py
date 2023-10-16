from abc import ABC, abstractmethod
import pandas as pd


class GeneralOptimization(ABC):
    """
    Abstract base class for portfolio optimization. Subclasses should implement the `optimize`, `clean_weights`, and `get_weights` methods.
    """

    def __init__(self, df: pd.DataFrame, mcaps=None):
        self.df = df
        self.mcaps = mcaps

    def apply_kwargs(self, kwargs):
        """
        Apply the kwargs to the optimization object.

        Args:
        -----
        kwargs : dict
            A dictionary containing the kwargs to be applied to the optimization object.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def get_weights(self) -> pd.Series:
        """
        Get the optimized portfolio weights.

        Returns:
        --------
        weights : pandas.Series
            A pandas Series object containing the optimized weights for each asset in the portfolio.
        """
        pass

    @abstractmethod
    def get_metrics(self):
        """
        Get the metrics, such as performances, for the optimized portfolio.

        Returns:
        --------
        metrics : dict
            A dictionary containing the metrics for the optimized portfolio.
        """
        pass
