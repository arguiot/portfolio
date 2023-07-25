import numpy as np
import pandas as pd
from portfolio_optimization.data_collection.get_crypto_price_range import (
    get_historical_prices_for_assets,
)
from portfolio_optimization.portfolio.Portfolio import Portfolio


class Backtest:
    def __init__(
        self,
        portfolio: Portfolio,
        start_date,
        end_date,
        rebalance_frequency="W",
        folder_path="./data/csv",
        data=None,
    ):
        self.price_data = (
            get_historical_prices_for_assets(portfolio.assets(), folder_path)
            if data is None
            else data
        )
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.rebalance_frequency = rebalance_frequency
        self.portfolio = portfolio
        self.portfolio_value = pd.DataFrame(
            index=self.price_data.index, columns=["Portfolio Value"]
        )
        # DataFrame with a date index and a weight column for each rebalance date
        self.portfolio_compositions = pd.Series(name="Weights")

    def run_backtest(self, look_back_period=4, look_back_freq="M"):
        total_dates = pd.date_range(start=self.start_date, end=self.end_date, freq="D")
        rebalance_dates = pd.date_range(
            start=self.start_date, end=self.end_date, freq=self.rebalance_frequency
        )

        for date in total_dates:
            # Get asset prices for the date
            prices = self.price_data.loc[date]

            # Rebalance the portfolio weekly
            if date in rebalance_dates:
                # Get the last 4 months of data up to the current date
                look_back_range = pd.date_range(
                    end=date, periods=look_back_period, freq=look_back_freq
                )
                historical_data = self.price_data.loc[look_back_range]
                try:
                    self.portfolio.rebalance(historical_data)
                except ValueError as e:
                    print(e)
                    print(
                        f"Skipping rebalance on {date.strftime('%Y-%m-%d')} due to insufficient data."
                    )
                    continue

                portfolio_weights = self.portfolio.weights
                self.portfolio_compositions.loc[date] = portfolio_weights

            # Calculate and log daily portfolio value
            self.portfolio_value.loc[date, "Portfolio Value"] = self.portfolio.value(
                prices
            )

        return self.portfolio_value, rebalance_dates, self.portfolio_compositions
