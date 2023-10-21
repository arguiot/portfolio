import pandas as pd
from typing import List
from ..backtesting import Backtest, PortfolioPerformance
from ..optimization import RiskParity


def optimal_strategy(backtests: List[List[PortfolioPerformance]]):
    """
    Returns the optimal strategy based on the backtests.
    """
    # Get 'Risk Parity' backtest
    portfolios = []
    for backtest in backtests:
        for portfolio in backtest:
            if portfolio.name == "Risk Parity":
                portfolios.append(portfolio)

    values = [portfolio.portfolio_value for portfolio in portfolios]
    df = pd.concat(values, axis=1)
    df.columns = [portfolio.name for portfolio in portfolios]

    rp = RiskParity(df)
    weights = rp.get_weights()

    return weights
