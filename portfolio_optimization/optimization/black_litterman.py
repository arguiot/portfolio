from .GeneralOptimization import GeneralOptimization
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import expected_returns
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting


class BlackLitterman(GeneralOptimization):
    def __init__(self, df, cov=None, weight_bounds=(0, 1)):
        """
        Initialize the Markowitz class.

        Parameters:
        -----------
        df : pandas.DataFrame
            A DataFrame of asset prices, where each column represents a different asset.
        """
        super().__init__(df)

        self.weight_bounds = weight_bounds

        if cov is None:
            self.cov_matrix = self.get_cov_matrix()
        else:
            self.cov_matrix = cov

        self.rets = expected_returns.mean_historical_return(df)

    def get_cov_matrix(self):
        """
        Get the covariance matrix for the given data.

        Returns:
        --------
        cov_matrix : pandas.DataFrame
            A pandas DataFrame object containing the covariance matrix for the given data.
        """
        return CovarianceShrinkage(self.df).ledoit_wolf()
