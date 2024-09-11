from abc import ABC, abstractmethod
import pandas as pd
import functools


class GeneralOptimizationDelegate:
    """
    Abstract base class for portfolio optimization delegates. Subclasses can implement the `optimize` method.
    """

    def setup(self, optimization_object):
        """
        Setup the optimization object. Method is called right before the optimization is performed.

        Returns:
        --------
        None
        """
        pass


# Return a function that, when called, will create a MyClass instance with the given kwargs. Then, is sets the delegate property of the instance to the delegate object.
def bind_delegate(cls, delegate, *args, **kwargs):
    instance = cls(*args, **kwargs)
    instance.delegate = delegate
    return instance


class GeneralOptimization(ABC):
    """
    Abstract base class for portfolio optimization. Subclasses should implement the `optimize`, `clean_weights`, and `get_weights` methods.
    """

    def __init__(self, df: pd.DataFrame, mcaps=None):
        self.df = df
        self.mcaps = mcaps
        self.delegate = GeneralOptimizationDelegate()

    # Static bind delegate method
    @classmethod
    def bind(cls, delegate: GeneralOptimizationDelegate):
        return functools.partial(bind_delegate, cls, delegate)

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

    def process_asset_weight_bounds(self, columns=None):
        assert hasattr(self, "asset_weight_bounds"), (
            "asset_weight_bounds attribute not found. "
            "Please make sure you have passed in the asset_weight_bounds argument "
            "when initializing the optimization object."
        )
        if self.asset_weight_bounds is None:
            self.asset_weight_bounds = {"*": (0, 1)}
        if "*" not in self.asset_weight_bounds:
            self.asset_weight_bounds["*"] = (0, 1)

        # Now, let's expand the asset_weight_bounds dictionary to include all assets in the dataframe
        # First, let's get all the assets in the dataframe
        assets = columns if columns is not None else self.df.columns
        # Now, let's expand the asset_weight_bounds dictionary to include all assets in the dataframe
        for asset in assets:
            if asset not in self.asset_weight_bounds:
                self.asset_weight_bounds[asset] = self.asset_weight_bounds["*"]
        # Now, let's remove the "*" key from the dictionary
        del self.asset_weight_bounds["*"]
