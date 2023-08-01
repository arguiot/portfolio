import numpy as np
import pandas as pd
from portfolio_optimization.data_collection.get_crypto_price_range import (
    get_historical_prices_for_assets,
)
from portfolio_optimization.portfolio.Portfolio import Portfolio

from typing import Dict, List
from pandas.core.frame import DataFrame, Series


class PortfolioPerformance:
    def __init__(
        self,
        portfolio_name: str,
        portfolio_value: DataFrame,
        rebalance_dates: Series,
        portfolio_compositions: Series,
        portfolio_holdings: Series,
        portfolio_metrics: Series = None,
    ):
        self.name = portfolio_name
        self.portfolio_value = portfolio_value
        self.rebalance_dates = rebalance_dates
        self.portfolio_compositions = portfolio_compositions
        self.portfolio_holdings = portfolio_holdings
        self.portfolio_metrics = portfolio_metrics


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
        self.portfolio_metrics = {
            name: pd.Series(name="Metrics") for name in portfolios.keys()
        }

    def run_backtest(self, look_back_period=4, look_back_unit="M"):
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
                    start = date - pd.to_timedelta(
                        look_back_period, unit=look_back_unit
                    )
                    historical_data = self.price_data.loc[start:date]
                    try:
                        mcaps = (
                            self.mcaps.loc[start:date]
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
                    self.portfolio_metrics[name].loc[date] = portfolio.get_metrics()

            portfolio_performances.append(
                PortfolioPerformance(
                    name,
                    self.portfolio_values[name],
                    rebalance_dates,
                    self.portfolio_compositions[name],
                    self.portfolio_holdings[name],
                    self.portfolio_metrics[name],
                )
            )

        return portfolio_performances

    def export_results(
        self, performances: List[PortfolioPerformance], folder_path: str
    ):
        """
        Export the results of the backtest to an Excel file.
        """
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(
            f"{folder_path}/backtest_results.xlsx", engine="xlsxwriter"
        )

        # For each portfolio, create a separate sheet.
        for performance in performances:
            # Write each DataFrame to a different worksheet.
            performance.portfolio_value.dropna().to_excel(
                writer, sheet_name=performance.name, startrow=3
            )
            print("performance.portfolio_metrics:")
            print(performance.portfolio_metrics)
            if performance.portfolio_metrics is not None:
                performance.portfolio_metrics = performance.portfolio_metrics.dropna()
                metrics = performance.portfolio_metrics.tolist()

                # ensuring that metrics is not an empty list
                if metrics:
                    metrics_df = pd.DataFrame(metrics)
                    metrics_df.index = performance.portfolio_metrics.index
                    metrics_df.to_excel(
                        writer, sheet_name=performance.name, startrow=3, startcol=5
                    )

            # Convert each item of the series to a DataFrame and then to excel
            compositions_df = pd.DataFrame(performance.portfolio_compositions.tolist())
            compositions_df.index = performance.portfolio_compositions.index
            compositions_df.to_excel(
                writer, sheet_name=performance.name, startrow=3, startcol=15
            )

            holdings__df = pd.DataFrame(performance.portfolio_holdings.tolist())
            holdings__df.index = performance.portfolio_holdings.index
            holdings__df.to_excel(
                writer, sheet_name=performance.name, startrow=3, startcol=30
            )

            # Write the portfolio name at the top of the sheet
            sheet = writer.sheets[performance.name]
            sheet.write(0, 0, performance.name)

            # Write headers for each DataFrame
            sheet.write(2, 0, "Portfolio Value")
            sheet.write(2, 5, "Portfolio Metrics")
            sheet.write(2, 15, "Portfolio Compositions")
            sheet.write(2, 30, "Portfolio Holdings")

            # Calculate end row and end column for creating chart
            end_row = performance.portfolio_value.shape[0] + 2
            end_col = performance.portfolio_value.shape[1] - 1

            # Create a new chart object
            chart = writer.book.add_chart({"type": "line"})

            # Add a series to the chart
            chart.add_series(
                {
                    "name": f"='{performance.name}'!$B$3",
                    "categories": f"='{performance.name}'!$A$4:$A${end_row}",
                    "values": f"='{performance.name}'!$B$4:$B${end_row}",
                }
            )

            # Insert the chart into the worksheet
            sheet.insert_chart("G10", chart)

        # Close the Pandas Excel writer and output the Excel file.
        writer.close()
