from pypfopt import expected_returns as er
def expected_returns(df):
    """
    Calculates the expected returns from a DataFrame of asset prices.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame of asset prices, where each column represents a different asset.

    Returns:
    --------
    pandas.Series
        A Series of expected returns for each asset, indexed by the asset names.
    """
    return er.returns_from_prices(df)