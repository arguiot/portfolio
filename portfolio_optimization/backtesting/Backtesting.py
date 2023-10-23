import pandas as pd
import numpy as np
import string
import traceback
from multiprocessing import Pool
from ..data_collection.get_crypto_price_range import (
    get_historical_prices_for_assets,
)
from ..portfolio.Portfolio import Portfolio

from typing import Dict, List
from pandas.core.frame import DataFrame, Series


class PortfolioPerformance:
    def __init__(
        self,
        portfolio_name: str,
        portfolio_value: DataFrame,
        rebalance_dates: Series,
        portfolio_compositions: Series,
        portfolio_raw_composition: Series,
        portfolio_holdings: Series,
        portfolio_metrics: Series = None,
    ):
        self.name = portfolio_name
        self.portfolio_value = portfolio_value
        self.rebalance_dates = rebalance_dates
        self.portfolio_compositions = portfolio_compositions
        self.portfolio_raw_composition = portfolio_raw_composition
        self.portfolio_holdings = portfolio_holdings
        self.portfolio_metrics = portfolio_metrics

    def decompose_grouped_tokens(self, rosetta: pd.Series):
        decomposed = pd.DataFrame()
        for token in self.portfolio_holdings.index:
            decomposed[token] = Portfolio.decompose_grouped_tokens(
                self.portfolio_holdings[token], rosetta
            )
        return decomposed


class Backtest:
    def __init__(
        self,
        portfolios: Dict[str, Portfolio],
        start_date,
        end_date,
        rebalance_frequency="W",
        adjust_holdings=True,
        folder_path="./data/csv",
        data=None,
        mcaps=None,
        asset_class=None,
        progress_logger=None,
    ):
        self.asset_class = asset_class
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
        self.adjust_holdings = adjust_holdings
        self.portfolios = portfolios
        self.portfolio_values = {
            name: pd.DataFrame(index=self.price_data.index, columns=["Portfolio Value"])
            for name in portfolios.keys()
        }
        self.portfolio_compositions = {
            name: pd.Series(name="Weights") for name in portfolios.keys()
        }
        self.portfolio_raw_composition = {
            name: pd.Series(name="Raw Weights") for name in portfolios.keys()
        }

        self.portfolio_holdings = {
            name: pd.Series(name="Holdings") for name in portfolios.keys()
        }
        self.portfolio_metrics = {
            name: pd.Series(name="Metrics") for name in portfolios.keys()
        }

        # For each portfolio, we add a task to the progress logger
        for name in portfolios.keys():
            progress_logger.add_task(name, 1) if progress_logger is not None else None

    def process_portfolio(
        self,
        name,
        portfolio: Portfolio,
        total_dates,
        rebalance_dates,
        look_back_period,
        look_back_unit,
        yield_data,
        progress_logger=None,
    ):
        for date in total_dates:
            prices = self.price_data.loc[date]
            # Apply the daily yield to the portfolio
            if yield_data is not None:
                portfolio.apply_yield(yield_data, compounded=True)
            self.portfolio_values[name].loc[date, "Portfolio Value"] = portfolio.value(
                prices
            )
            if self.adjust_holdings:
                portfolio.match_weights(prices)
            if date in rebalance_dates:
                start = date - pd.to_timedelta(look_back_period, unit=look_back_unit)
                historical_data = self.price_data.loc[start:date]
                try:
                    mcaps = self.mcaps.loc[date] if self.mcaps is not None else None

                    portfolio.rebalance(
                        historical_data,
                        prices,
                        self.portfolio_values[name].loc[date, "Portfolio Value"],
                        mcaps,
                    )
                except Exception as e:
                    print(str(e))
                    traceback.print_exc()
                    print(
                        f"Skipping rebalance for {name} on {date.strftime('%Y-%m-%d')} due to insufficient data."
                    )

                self.portfolio_compositions[name].loc[date] = portfolio.weights
                self.portfolio_raw_composition[name].loc[date] = portfolio.raw_weights
                self.portfolio_metrics[name].loc[date] = portfolio.get_metrics()
            self.portfolio_holdings[name].loc[date] = portfolio.holdings
        progress_logger.end_task(name) if progress_logger is not None else None

        return PortfolioPerformance(
            name,
            self.portfolio_values[name],
            rebalance_dates,
            self.portfolio_compositions[name],
            self.portfolio_raw_composition[name],
            self.portfolio_holdings[name],
            self.portfolio_metrics[name],
        )

    def run_backtest(
        self,
        look_back_period=4,
        look_back_unit="M",
        yield_data=None,
        progress_logger=None,
    ):
        total_dates = pd.date_range(start=self.start_date, end=self.end_date, freq="D")
        rebalance_dates = pd.date_range(
            start=self.start_date, end=self.end_date, freq=self.rebalance_frequency
        )

        portfolio_performances = []

        # if there is only one portfolio, we don't need to use multiprocessing
        if len(self.portfolios) == 1:
            for name, portfolio in self.portfolios.items():
                portfolio_performances.append(
                    self.process_portfolio(
                        name,
                        portfolio,
                        total_dates,
                        rebalance_dates,
                        look_back_period,
                        look_back_unit,
                        yield_data,
                        progress_logger,
                    )
                )
        else:
            with Pool() as p:
                portfolio_performances = p.starmap(
                    self.process_portfolio,
                    [
                        (
                            name,
                            portfolio,
                            total_dates,
                            rebalance_dates,
                            look_back_period,
                            look_back_unit,
                            yield_data,
                            progress_logger,
                        )
                        for name, portfolio in self.portfolios.items()
                    ],
                )

        return portfolio_performances

    def portfolio_analysis(
        self, df: pd.DataFrame | pd.Series, name, risk_free_rate_pos=11, is_asset=False
    ):
        prices = df[name].dropna() if is_asset else df.dropna()
        daily_return = prices.pct_change()

        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1

        average_daily_return = daily_return.mean()
        average_monthly_return = ((1 + average_daily_return) ** 30) - 1
        apy = ((1 + average_daily_return) ** 365) - 1
        cagr = ((prices.iloc[-1] / prices.iloc[0]) ** (1 / (len(prices) / 365))) - 1

        letter_index = (
            string.ascii_uppercase[df.columns.get_loc(name)] if is_asset is True else 0
        )
        key = (
            "'" + name + "'!C:C"
            if not is_asset
            else f"'Price Data'!{letter_index}:{letter_index}"
        )

        sharpe_ratio_formula = (
            f"=(AVERAGE({key})/STDEV({key})-$B${risk_free_rate_pos})*SQRT(365)"
        )

        volatility = daily_return.std()
        annualized_volatility = volatility * np.sqrt(365)

        # Maximum drawdown
        running_max = np.maximum.accumulate(prices)
        running_max[running_max < 1] = 1

        drawdown = prices / running_max - 1.0
        max_dd = drawdown.min()

        calmar_ratio = f"={cagr}-B{risk_free_rate_pos}/{max_dd}"

        return daily_return, pd.Series(
            {
                "Total Return": total_return,
                "Average Daily Return": average_daily_return,
                "Average Monthly Return": average_monthly_return,
                "Average Annual Return": apy,
                "CAGR": cagr,
                "Sharpe Ratio": sharpe_ratio_formula,
                "Daily Volatility": volatility,
                "Annualized Volatility": annualized_volatility,
                "Max Drawdown": max_dd,
                "Calmar Ratio": calmar_ratio,
            }
        )

    def export_results(
        self,
        performances: List[PortfolioPerformance],
        folder_path: str,
        file_name: str = "backtest_results.xlsx",
    ):
        """
        Export the results of the backtest to an Excel file.
        """
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(f"{folder_path}/{file_name}", engine="xlsxwriter")

        # Copy price_data and crop it to the start and end dates of the backtest
        price_data = self.price_data.copy()
        price_data = price_data.loc[self.start_date : self.end_date]

        price_data.to_excel(writer, sheet_name="Price Data", index=True)
        writer.sheets["Price Data"].hide()

        overview_df = pd.DataFrame(
            index=[
                "Total Return",
                "Average Daily Return",
                "Average Monthly Return",
                "Average Annual Return",
                "CAGR",
                "Sharpe Ratio",
                "Daily Volatility",
                "Annualized Volatility",
                "Max Drawdown",
                "Calmar Ratio",
            ]
        )

        overview_df.to_excel(
            writer, sheet_name="Overview", index=True
        )  # Making sure the sheet is the first one

        # For each portfolio, create a separate sheet.
        for performance in performances:
            # Write each DataFrame to a different worksheet.
            value = performance.portfolio_value.dropna()
            # Calculate the metrics
            value = performance.portfolio_value.dropna()
            # Add to dataframe
            risk_free_rate_pos = overview_df.shape[0] + 4
            daily_return, overview_df[performance.name] = self.portfolio_analysis(
                value["Portfolio Value"], performance.name, risk_free_rate_pos
            )
            value["Daily Return"] = daily_return
            value.to_excel(writer, sheet_name=performance.name, startrow=3)

            # Write the portfolio name at the top of the sheet
            sheet = writer.sheets[performance.name]
            sheet.write(0, 0, performance.name)

            sheet.write(2, 0, "Portfolio Value")

            # Convert each item of the series to a DataFrame and then to excel
            sheet.write(2, 5, "Portfolio Weights")
            compositions_df = pd.DataFrame(performance.portfolio_compositions.tolist())
            compositions_df.index = performance.portfolio_compositions.index
            compositions_df.to_excel(
                writer, sheet_name=performance.name, startrow=3, startcol=5
            )

            startcol = 10 + compositions_df.shape[1]
            sheet.write(2, startcol, "Portfolio Raw Weights")
            raw_compositions_df = pd.DataFrame(
                performance.portfolio_raw_composition.tolist()
            )
            raw_compositions_df.index = performance.portfolio_raw_composition.index
            raw_compositions_df.to_excel(
                writer, sheet_name=performance.name, startrow=3, startcol=startcol
            )

            holdings__df = pd.DataFrame(performance.portfolio_holdings.tolist())
            holdings__df.index = performance.portfolio_holdings.index
            # Start col is 5 after end of compositions
            startcol += 5 + compositions_df.shape[1]
            sheet.write(2, startcol, "Portfolio Composition")
            holdings__df.to_excel(
                writer, sheet_name=performance.name, startrow=3, startcol=startcol
            )

            # Calculate end row and end column for creating chart
            end_row = performance.portfolio_value.shape[0] + 2
            end_col = performance.portfolio_value.shape[1] - 1

        overview_df.to_excel(writer, sheet_name="Overview", index=True)

        # Erease the content of `overview_df` and redo for assets
        overview_df = pd.DataFrame()

        for asset in self.price_data.columns:
            try:
                overview_df[asset] = self.portfolio_analysis(
                    self.price_data, asset, risk_free_rate_pos, is_asset=True
                )[1]
            except Exception as e:
                print(str(e))
                traceback.print_exc()
                print(
                    f"Skipping analysis for {asset} due to insufficient data. Make sure the price data is complete."
                )

        overview_df.to_excel(
            writer,
            sheet_name="Overview",
            index=True,
            startrow=overview_df.shape[0] + 23,
        )

        # Set the "Overview" sheet as the active sheet.
        writer.sheets["Overview"].activate()
        writer.sheets["Overview"].set_first_sheet()

        # Create a dropdown list selector for the charts
        chart_sheet = writer.sheets["Overview"]

        chart_sheet.write(overview_df.shape[0] + 3, 0, "Risk Free Rate")
        chart_sheet.write(overview_df.shape[0] + 3, 1, 0.02)
        chart_sheet.write(overview_df.shape[0] + 5, 1, performances[0].name)
        chart_sheet.data_validation(
            f"B{overview_df.shape[0] + 6}",
            {
                "validate": "list",
                "source": [portfolio.name for portfolio in performances],
            },
        )

        # Create a hidden sheet for the chart data
        writer.sheets["Chart Data"] = writer.book.add_worksheet("Chart Data")
        writer.sheets["Chart Data"].hide()

        # Use =INDIRECT("'"&INDEX(Overview!B1:H1,1,Overview!B10)&"'!A5:"&"C"&COUNTA(INDIRECT("'"&INDEX(Overview!B1:H1,1,Overview!B10)&"'!A:A")))
        writer.sheets["Chart Data"].write_dynamic_array_formula(
            "A1:A1",
            f'=INDIRECT("\'"&Overview!B{overview_df.shape[0] + 6}&"\'!A5:"&"C"&COUNTA(INDIRECT("\'"&Overview!B{overview_df.shape[0] + 6}&"\'!A:A")))',
        )

        # Format A column as dates
        date_format = writer.book.add_format({"num_format": "yyyy-mm-dd"})
        writer.sheets["Chart Data"].set_column("A:A", None, date_format)

        # Create a chart object
        chart = writer.book.add_chart({"type": "line"})
        chart.set_title({"name": "Portfolio Value"})
        chart.set_x_axis({"name": "Date"})
        chart.set_y_axis({"name": "Portfolio Value"})
        chart.add_series(
            {
                "categories": "='Chart Data'!$A$1:$A$1000",
                "values": "='Chart Data'!$B$1:$B$1000",
                "name": "=Overview!$B$13",
            }
        )

        # Insert the chart into the sheet
        chart_sheet.insert_chart(f"B{overview_df.shape[0] + 8}", chart)
        # Close the Pandas Excel writer and output the Excel file.
        writer.close()
