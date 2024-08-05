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
from pandas.core.frame import DataFrame
from pandas import Series
from .delegate import BacktestingDelegate
from datetime import datetime
from pypfopt import expected_returns


class PortfolioPerformance:

    def __init__(
        self,
        portfolio_name: str,
        portfolio_value: DataFrame,
        rebalance_dates: Series,
        portfolio_compositions: Series,
        portfolio_raw_composition: Series,
        portfolio_holdings: Series,
        portfolio_live_weights: Series,
        portfolio_metrics: Series | None = None,
    ):
        self.name = portfolio_name
        self.portfolio_value = portfolio_value
        self.rebalance_dates = rebalance_dates
        self.portfolio_compositions = portfolio_compositions
        self.portfolio_raw_composition = portfolio_raw_composition
        self.portfolio_holdings = portfolio_holdings
        self.portfolio_live_weights = portfolio_live_weights
        self.portfolio_metrics = portfolio_metrics

    def decompose_grouped_tokens(self, rosetta: pd.Series):
        decomposed = pd.DataFrame()
        for token in self.portfolio_holdings.index:
            decomposed[token] = Portfolio.decompose_grouped_tokens(  # type: ignore
                self.portfolio_holdings[token], rosetta
            )
        return decomposed

    def up_to(
        self, end_date: datetime | None = None, look_back_period: int | None = None
    ):
        """
        Returns a copy of the PortfolioPerformance object up to the specified end date.
        """
        if end_date is None:
            return self
        end_date = pd.to_datetime(end_date)
        start_date = (
            end_date - pd.to_timedelta(look_back_period, unit="D")
            if look_back_period is not None
            else self.rebalance_dates[0]
        )
        return PortfolioPerformance(
            self.name,
            self.portfolio_value.loc[start_date:end_date],
            self.rebalance_dates[
                (self.rebalance_dates >= start_date)
                & (self.rebalance_dates <= end_date)
            ],
            self.portfolio_compositions.loc[start_date:end_date],
            self.portfolio_raw_composition.loc[start_date:end_date],
            self.portfolio_holdings.loc[start_date:end_date],
            self.portfolio_live_weights.loc[start_date:end_date],
            (
                self.portfolio_metrics.loc[start_date:end_date]
                if self.portfolio_metrics is not None
                else None
            ),
        )

    def starting_from(self, start_date: datetime | None = None):
        """
        Returns a copy of the PortfolioPerformance object starting from the specified start date.
        """
        if start_date is None:
            return self
        start_date = pd.to_datetime(start_date)
        return PortfolioPerformance(
            self.name,
            self.portfolio_value.loc[start_date:],
            self.rebalance_dates[self.rebalance_dates >= start_date],
            self.portfolio_compositions.loc[start_date:],
            self.portfolio_raw_composition.loc[start_date:],
            self.portfolio_holdings.loc[start_date:],
            self.portfolio_live_weights.loc[start_date:],
            (
                self.portfolio_metrics.loc[start_date:]
                if self.portfolio_metrics is not None
                else None
            ),
        )

    def scale_at_date(self, date: datetime, value: float):
        """
        Scales the portfolio value at a given date, so that, at that date, the portfolio value is equal to the given value.
        """
        if date is None:
            return self

        value_at_date = self.portfolio_value.loc[date]
        scale_factor = value / value_at_date

        print(f"Scaling {self.name} by {scale_factor} at {date}")

        # Scale value & holdings
        self.portfolio_value *= scale_factor
        self.portfolio_holdings *= scale_factor["Portfolio Value"]
        # print(f"*Holdings* {self.portfolio_holdings}")

        print(f"New value: {self.portfolio_value.loc[date]}")

        return self


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
        self.delegate = BacktestingDelegate()
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

        self.portfolio_live_weights = {
            name: pd.Series(name="Live Weights") for name in portfolios.keys()
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
                    _current_value = portfolio.value(prices)
                    portfolio.rebalance(
                        historical_data,
                        prices,
                        _current_value,
                        mcaps,
                        yield_data,
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
            self.portfolio_holdings[name].loc[date] = pd.Series(
                portfolio.holdings, copy=False
            )
            self.portfolio_live_weights[name].loc[date] = (
                portfolio.holdings * prices / (portfolio.value(prices))
            )
            self.delegate.post_process(self, portfolio, date)
        (
            progress_logger.end_task(name)
            if progress_logger is not None and hasattr(progress_logger, "end_task")
            else None
        )

        return PortfolioPerformance(
            name,
            self.portfolio_values[name],
            rebalance_dates,
            self.portfolio_compositions[name],
            self.portfolio_raw_composition[name],
            self.portfolio_holdings[name],
            self.portfolio_live_weights[name],
            self.portfolio_metrics[name],
        )

    # Utils
    def returns_from_prices(self, prices, log_returns=False):
        """
        Calculate the returns given prices.

        :param prices: adjusted (daily) closing prices of the asset, each row is a
                    date and each column is a ticker/id.
        :type prices: pd.DataFrame
        :param log_returns: whether to compute using log returns
        :type log_returns: bool, defaults to False
        :return: (daily) returns
        :rtype: pd.DataFrame
        """
        if log_returns:
            returns = np.log(1 + prices.pct_change())
        else:
            returns = prices.pct_change()
        return returns

    def prices_from_returns(self, returns, log_returns=False):
        """
        Calculate the pseudo-prices given returns. These are not true prices because
        the initial prices are all set to 1, but it behaves as intended when passed
        to any PyPortfolioOpt method.

        :param returns: (daily) percentage returns of the assets
        :type returns: pd.DataFrame
        :param log_returns: whether to compute using log returns
        :type log_returns: bool, defaults to False
        :return: (daily) pseudo-prices.
        :rtype: pd.DataFrame
        """
        if log_returns:
            ret = np.exp(returns)
        else:
            ret = 1 + returns
        ret.iloc[0] = 1  # set first day pseudo-price
        return ret.cumprod()

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

        if yield_data is not None:
            returns = self.returns_from_prices(self.price_data, log_returns=False)

            for asset in yield_data.index:
                if asset not in self.price_data.columns:
                    continue
                daily_yield = (1 + yield_data[asset]) ** (1 / 365) - 1

                # Shift the yield by one day
                shifted_yield = pd.Series(daily_yield, index=returns.index).shift(1)
                returns[asset] = returns[asset] + shifted_yield

            # Remove the first row of returns as it now contains NaN from the shift
            returns = returns.iloc[1:]

            # Calculate cumulative returns
            cumulative_returns = (1 + returns).cumprod()

            # Initialize new_prices with the same structure as self.price_data
            new_prices = pd.DataFrame(
                index=self.price_data.index, columns=self.price_data.columns
            )

            for asset in self.price_data.columns:
                # Find the first non-NaN price for each asset
                first_valid_index = self.price_data[asset].first_valid_index()
                if first_valid_index is not None:
                    initial_price = self.price_data.loc[first_valid_index, asset]

                    # Calculate new prices using the initial price and cumulative returns
                    asset_cumulative_returns = cumulative_returns[asset].reindex(
                        new_prices.index
                    )
                    new_prices[asset] = asset_cumulative_returns * initial_price

                    # Preserve NaN values at the beginning
                    new_prices.loc[:first_valid_index, asset] = self.price_data.loc[
                        :first_valid_index, asset
                    ]

            self.price_data = new_prices
            # Rebalance the portfolios
            for name, portfolio in self.portfolios.items():
                start = self.start_date
                dataf = self.price_data.loc[:start]
                current_prices = dataf.iloc[-1]

                print(f"Rebalancing {name}")
                print(f"Current Prices: {current_prices}")
                print(f"Base Value: {portfolio.base_value}")
                print(f"DataFrame: {dataf}")

                portfolio.rebalance(
                    df=dataf,
                    current_prices=current_prices,
                    base_value=portfolio.base_value,
                    mcaps=self.mcaps,
                    yield_data=yield_data,
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
        # APY is total return, divided by the number of years, plus 1, raised to the power of 1 divided by the number of years, minus 1
        start_date = prices.index[0]
        end_date = prices.index[-1]
        years = (end_date - start_date).days / 365
        apy = ((1 + total_return) ** (1 / years)) - 1
        cagr = ((prices.iloc[-1] / prices.iloc[0]) ** (1 / (len(prices) / 365))) - 1

        # Convert to DataFrame if df is a pd.Series
        if is_asset is True:
            if isinstance(df, pd.Series):
                df = df.to_frame()

            columns = df.columns.to_list()
            # Find name in the columns and get the letter index
            try:
                column_index = columns.index(name) + len(columns)
            except ValueError:
                print(
                    f"Could not find {name} in the columns of the dataframe. Here are the columns: {columns}"
                )
                raise ValueError
            if (column_index + 1) < 26:
                letter_index = string.ascii_uppercase[column_index + 1]
            else:
                quotient, remainder = divmod(column_index + 1, 26)
                letter_index = (
                    string.ascii_uppercase[quotient - 1]
                    + string.ascii_uppercase[remainder]
                )
        else:
            letter_index = 0
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
                "90d Volatility": daily_return.rolling(90).std().iloc[-1],
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
        additional_data: Dict[str, pd.DataFrame] | None = None,
    ):
        """
        Export the results of the backtest to an Excel file.
        """
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(f"{folder_path}/{file_name}", engine="xlsxwriter")

        # Copy price_data and crop it to the start and end dates of the backtest
        price_data = self.price_data.copy()
        price_data = price_data.loc[self.start_date : self.end_date]
        # Add all the daily returns to the price data
        for asset in price_data.columns:
            price_data[asset + " Daily Return"] = price_data[asset].pct_change()
            # Add market cap if available
            if self.mcaps is not None and asset in self.mcaps.columns:
                price_data[asset + " Market Cap"] = self.mcaps[asset]

        price_data.sort_index(inplace=True)

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
                "90d Volatility",
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

            # Live weight is the basically the holding * price / portfolio value.
            live_weights_df = pd.DataFrame(performance.portfolio_live_weights.tolist())
            live_weights_df.index = performance.portfolio_live_weights.index
            startcol += 5 + live_weights_df.shape[1]
            sheet.write(2, startcol, "Live Weight")
            live_weights_df.to_excel(
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
                    self.price_data.loc[self.start_date : self.end_date],
                    asset,
                    risk_free_rate_pos,
                    is_asset=True,
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
            f'=INDIRECT("\'"&Overview!B{overview_df.shape[0] + 6}&"\'!A5:"&"C"&COUNTA(INDIRECT("\'"&Overview!B{overview_df.shape[0] + 6}&"\'!A:A"))+2)',
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
                "categories": "='Chart Data'!$A:$A",
                "values": "='Chart Data'!$B:$B",
                "name": "=Overview!$B$13",
            }
        )

        # Insert the chart into the sheet
        chart_sheet.insert_chart(f"B{overview_df.shape[0] + 8}", chart)
        # Close the Pandas Excel writer and output the Excel file.
        writer.close()
