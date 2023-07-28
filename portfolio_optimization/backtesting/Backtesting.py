import numpy as np
import pandas as pd
from portfolio_optimization.data_collection.get_crypto_price_range import (
    get_historical_prices_for_assets,
)
from portfolio_optimization.portfolio.Portfolio import Portfolio

from typing import Dict, Tuple
from pandas.core.frame import DataFrame, Series


class PortfolioPerformance:
    def __init__(
        self,
        portfolio_name: str,
        portfolio_value: DataFrame,
        rebalance_dates: Series,
        portfolio_compositions: Series,
        portfolio_holdings: Series,
    ):
        self.name = portfolio_name
        self.portfolio_value = portfolio_value
        self.rebalance_dates = rebalance_dates
        self.portfolio_compositions = portfolio_compositions
        self.portfolio_holdings = portfolio_holdings


class Backtest:
    def __init__(
        self,
        portfolios: Dict[str, Portfolio],
        start_date,
        end_date,
        rebalance_frequency="W",
        folder_path="./data/csv",
        data=None,
        mcaps=None,
    ):
        self.price_data = (
            get_historical_prices_for_assets(
                [
                    asset
                    for portfolio in portfolios.values()
                    for asset in portfolios.assets()
                ],
                folder_path,
            )
            if data is None
            else data
        )
        self.mcaps = mcaps
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.rebalance_frequency = rebalance_frequency
        self.portfolios = portfolios
        self.portfolio_values = {
            name: pd.DataFrame(index=self.price_data.index, columns=["Portfolio Value"])
            for name in portfolios.keys()
        }
        self.portfolio_compositions = {
            name: pd.Series(name="Weights") for name in portfolios.keys()
        }
        self.portfolio_holdings = {
            name: pd.Series(name="Holdings") for name in portfolios.keys()
        }

    def run_backtest(self, look_back_period=4, look_back_freq="M"):
        total_dates = pd.date_range(start=self.start_date, end=self.end_date, freq="D")
        rebalance_dates = pd.date_range(
            start=self.start_date, end=self.end_date, freq=self.rebalance_frequency
        )

        portfolio_performances = []

        for name, portfolio in self.portfolios.items():
            for date in total_dates:
                prices = self.price_data.loc[date]
                self.portfolio_values[name].loc[
                    date, "Portfolio Value"
                ] = portfolio.value(prices)
                if date in rebalance_dates:
                    look_back_range = pd.date_range(
                        end=date, periods=look_back_period, freq=look_back_freq
                    )
                    historical_data = self.price_data.loc[look_back_range]
                    try:
                        mcaps = (
                            self.mcaps.loc[look_back_range]
                            if self.mcaps is not None
                            else None
                        )

                        portfolio.rebalance(
                            historical_data,
                            prices,
                            self.portfolio_values[name].loc[date, "Portfolio Value"],
                            mcaps,
                        )
                    except ValueError as e:
                        print(e)
                        print(
                            f"Skipping rebalance on {date.strftime('%Y-%m-%d')} due to insufficient data."
                        )

                    self.portfolio_compositions[name].loc[date] = portfolio.weights
                    self.portfolio_holdings[name].loc[date] = portfolio.holdings

            portfolio_performances.append(
                PortfolioPerformance(
                    name,
                    self.portfolio_values[name],
                    rebalance_dates,
                    self.portfolio_compositions[name],
                    self.portfolio_holdings[name],
                )
            )

        return portfolio_performances
