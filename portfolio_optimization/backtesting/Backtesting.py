import pandas as pd
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
        self.portfolio_raw_composition = {
            name: pd.Series(name="Raw Weights") for name in portfolios.keys()
        }

        self.portfolio_holdings = {
            name: pd.Series(name="Holdings") for name in portfolios.keys()
        }
        self.portfolio_metrics = {
            name: pd.Series(name="Metrics") for name in portfolios.keys()
        }

    def process_portfolio(
        self,
        name,
        portfolio,
        total_dates,
        rebalance_dates,
        look_back_period,
        look_back_unit,
        yield_data,
    ):
        for date in total_dates:
            prices = self.price_data.loc[date]
            # Apply the daily yield to the portfolio
            if yield_data is not None:
                portfolio.apply_yield(yield_data, compounded=True)
            self.portfolio_values[name].loc[date, "Portfolio Value"] = portfolio.value(
                prices
            )
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
                    print(e)
                    print(
                        f"Skipping rebalance on {date.strftime('%Y-%m-%d')} due to insufficient data."
                    )

                self.portfolio_compositions[name].loc[date] = portfolio.weights
                self.portfolio_raw_composition[name].loc[date] = portfolio.raw_weights
                self.portfolio_holdings[name].loc[date] = portfolio.holdings
                self.portfolio_metrics[name].loc[date] = portfolio.get_metrics()
        print(f"Finished processing {name}")
        return PortfolioPerformance(
            name,
            self.portfolio_values[name],
            rebalance_dates,
            self.portfolio_compositions[name],
            self.portfolio_raw_composition[name],
            self.portfolio_holdings[name],
            self.portfolio_metrics[name],
        )

    def run_backtest(self, look_back_period=4, look_back_unit="M", yield_data=None):
        total_dates = pd.date_range(start=self.start_date, end=self.end_date, freq="D")
        rebalance_dates = pd.date_range(
            start=self.start_date, end=self.end_date, freq=self.rebalance_frequency
        )

        portfolio_performances = []

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
                    )
                    for name, portfolio in self.portfolios.items()
                ],
            )

        return portfolio_performances

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

        overview_df = pd.DataFrame(
            index=[
                "Total Return",
                "Average Daily Return",
                "Average Monthly Return",
                "Average Annual Return",
                "CAGR",
                "Sharpe Ratio",
                "Volatility",
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
            value["Daily Return"] = value["Portfolio Value"].pct_change()

            total_return = (
                value["Portfolio Value"].iloc[-1] / value["Portfolio Value"].iloc[0] - 1
            )

            average_daily_return = value["Daily Return"].mean()
            average_monthly_return = ((1 + average_daily_return) ** 30) - 1
            apy = ((1 + average_daily_return) ** 365) - 1
            cagr = (
                (value["Portfolio Value"].iloc[-1] / value["Portfolio Value"].iloc[0])
                ** (1 / (len(value) / 365))
            ) - 1
            # sharpe_ratio = (
            #     value["Daily Return"].mean() / value["Daily Return"].std()
            # ) * np.sqrt(365)
            sharpe_ratio_formula = f"=(AVERAGE('{performance.name}'!C:C)/STDEV('{performance.name}'!C:C)-$B$11)*SQRT(365)"

            volatility = value["Daily Return"].std()

            # Add to dataframe
            overview_df[performance.name] = pd.Series(
                {
                    "Total Return": total_return,
                    "Average Daily Return": average_daily_return,
                    "Average Monthly Return": average_monthly_return,
                    "Average Annual Return": apy,
                    "CAGR": cagr,
                    "Sharpe Ratio": sharpe_ratio_formula,
                    "Volatility": volatility,
                }
            )
            value.to_excel(writer, sheet_name=performance.name, startrow=3)

            # Write the portfolio name at the top of the sheet
            sheet = writer.sheets[performance.name]
            sheet.write(0, 0, performance.name)

            sheet.write(2, 0, "Portfolio Value")
            # if performance.portfolio_metrics is not None:
            #     performance.portfolio_metrics = performance.portfolio_metrics.dropna()
            #     metrics = performance.portfolio_metrics.tolist()

            #     # ensuring that metrics is not an empty list
            #     if metrics:
            #         metrics_df = pd.DataFrame(metrics)
            #         metrics_df.index = performance.portfolio_metrics.index
            #         metrics_df.to_excel(
            #             writer, sheet_name=performance.name, startrow=3, startcol=5
            #         )

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
        # Set the "Overview" sheet as the active sheet.
        writer.sheets["Overview"].activate()
        writer.sheets["Overview"].set_first_sheet()

        # Create a dropdown list selector for the charts
        chart_sheet = writer.sheets["Overview"]

        chart_sheet.write(10, 0, "Risk Free Rate")
        chart_sheet.write(10, 1, 0.02)
        chart_sheet.write(12, 1, performances[0].name)
        chart_sheet.data_validation(
            "B13",
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
            '=INDIRECT("\'"&Overview!B13&"\'!A5:"&"C"&COUNTA(INDIRECT("\'"&Overview!B13&"\'!A:A")))',
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
        chart_sheet.insert_chart("B15", chart)
        # Close the Pandas Excel writer and output the Excel file.
        writer.close()
