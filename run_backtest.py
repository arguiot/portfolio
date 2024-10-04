import pandas as pd
from portfolio_optimization.data_processing import *
from portfolio_optimization.data_collection import *
from tokens.get_assets import *
import sys
from portfolio_optimization.utils import ProgressLogger

from portfolio_optimization.backtesting.Backtesting import Backtest
from portfolio_optimization.backtesting.parity import (
    ParityLine,
    ParityBacktestingProcessor,
    ParityProcessorDelegate,
)
from main_backtest.create_portfolios import create_portfolios

np.random.seed(1)

# Define scenarios
scenarios = {
    # "stress_1": {
    #     "start_date": pd.to_datetime("2020-06-26"),
    #     "end_date": pd.to_datetime("2024-06-26"),
    # },
    "stress_2": {
        "start_date": pd.to_datetime("2021-06-26"),
        "end_date": pd.to_datetime("2024-06-26"),
    },
    # "bull_market": {
    #     "start_date": pd.to_datetime("2020-05-15"),
    #     "end_date": pd.to_datetime("2021-11-12"),
    # },
    # "bear_market": {
    #     "start_date": pd.to_datetime("2021-11-12"),
    #     "end_date": pd.to_datetime("2022-11-12"),
    # },
    # "sprint_bull_market": {
    #     "start_date": pd.to_datetime("2023-06-26"),
    #     "end_date": pd.to_datetime("2024-06-26"),
    # },
}

lookback_period = 120  # 120 days

parity_lookback_period = 90  # 90 days

daily_trade_generator = False

initial_cash = 10000.0


def run_for_asset_class(
    backtest: Backtest,
    rebalance_frequency="1D",
    progress_logger=None,
    scenario_tag=None,
    parity_lookback_period=parity_lookback_period,
    asset_list: dict[str, list[str]] = {},
    asset_class: str = "",
):
    yield_data = pd.Series()
    for asset in asset_list[asset_class]:
        if asset_class == "high_risk_tickers":
            yield_data[asset] = 0  # 0.075
        elif asset_class == "medium_risk_tickers":
            yield_data[asset] = 0
        elif asset_class == "low_risk_tickers":
            if asset in ["usdc", "usdt", "dai"]:
                yield_data[asset] = 0.07
            if asset == "frax":
                yield_data[asset] = 0.07
            if asset == "gho":
                yield_data[asset] = 0.06
            if asset == "fdusd":
                yield_data[asset] = 0.03
            if asset == "usde":
                yield_data[asset] = 0.20
            if asset == "usdd":
                yield_data[asset] = 0.06
            if asset == "crvusd":
                yield_data[asset] = 0.12
            if asset == "susd":
                yield_data[asset] = 0.06
            if asset == "tusd":
                yield_data[asset] = 0.10
            if asset == "pyusd":
                yield_data[asset] = 0.04
            if asset == "paxg":
                yield_data[asset] = 0.08
            if asset == "xaut":
                yield_data[asset] = 0.0
            if asset == "btc":
                yield_data[asset] = 0.0
            if asset == "eth":
                yield_data[asset] = 0.07
            if asset == "bnb":
                yield_data[asset] = 0.01
            if asset == "matic":
                yield_data[asset] = 0.04

    perfs = backtest.run_backtest(
        look_back_period=lookback_period,
        look_back_unit="D",
        yield_data=yield_data,
        progress_logger=progress_logger,
    )

    if not os.path.exists(f"./out/{rebalance_frequency}/"):
        os.makedirs(f"./out/{rebalance_frequency}/")

    tag = f"{scenario_tag}_lookback_{lookback_period}"
    filename = f"backtest_results_{asset_class}_{tag}.xlsx"

    # Truncate the backtest to the start date
    _perfs = list(
        map(
            lambda perf: perf.starting_from(
                perf.rebalance_dates[0] + pd.Timedelta(days=parity_lookback_period)
            ).scale_at_date(
                perf.rebalance_dates[0] + pd.Timedelta(days=parity_lookback_period),
                initial_cash,
            ),
            perfs,
        )
    )

    backtest.export_results(
        _perfs,
        f"./out/{rebalance_frequency}/",
        filename,
        export_csv=("--csv" in sys.argv),
    )

    return list(
        map(
            lambda perf: perf.scale_at_date(
                perf.rebalance_dates[0] + pd.Timedelta(days=parity_lookback_period),
                initial_cash,
            ),
            perfs,
        )
    )


def main(rebalance_frequency="1M", asset_class=None, csv_export=False):
    import warnings
    from typing import List, Tuple
    from portfolio_optimization.backtesting import PortfolioPerformance

    warnings.filterwarnings("ignore")

    progress_logger = ProgressLogger()
    asset_list = get_tickers()

    all_parity_perfs: List[PortfolioPerformance] = []
    all_asset_class_results: List[List[PortfolioPerformance]] = []

    for scenario, dates in scenarios.items():
        print(f"Running scenario: {scenario}")
        start_date = dates["start_date"] - pd.Timedelta(days=parity_lookback_period)
        end_date = dates["end_date"]

        if asset_class is not None:
            backtest = create_portfolios(
                asset_list,
                asset_class=asset_class,
                progress_logger=progress_logger.context(f"{scenario}_{asset_class}"),
                start_date=start_date,
                end_date=end_date,
                lookback_period=lookback_period,
                rebalance_frequency=rebalance_frequency,
                daily_trade_generator=daily_trade_generator,
                initial_cash=initial_cash,
            )
            results = run_for_asset_class(
                backtest=backtest,
                rebalance_frequency=rebalance_frequency,
                progress_logger=progress_logger,
                scenario_tag=scenario,
                parity_lookback_period=parity_lookback_period,
                asset_list=asset_list,
                asset_class=asset_class,
            )
            all_asset_class_results.append(results)
        else:
            asset_classes = [
                "high_risk_tickers",
                "medium_risk_tickers",
                "low_risk_tickers",
            ]
            backtests = []
            loggers = []

            for asset_class in asset_classes:
                logger = progress_logger.context(f"{scenario}_{asset_class}")
                backtest = create_portfolios(
                    asset_list,
                    asset_class=asset_class,
                    progress_logger=logger,
                    start_date=start_date,
                    end_date=end_date,
                    lookback_period=lookback_period,
                    rebalance_frequency=rebalance_frequency,
                    daily_trade_generator=daily_trade_generator,
                    initial_cash=initial_cash,
                )
                backtests.append(backtest)
                loggers.append(logger)

            perfs = []
            for backtest in backtests:
                asset_class = backtest.asset_class
                logger = loggers[backtests.index(backtest)]
                results = run_for_asset_class(
                    backtest=backtest,
                    rebalance_frequency=rebalance_frequency,
                    progress_logger=logger,
                    scenario_tag=scenario,
                    parity_lookback_period=parity_lookback_period,
                    asset_list=asset_list,
                    asset_class=asset_class,
                )
                all_asset_class_results.append(results)
                risk_parity = next(
                    result for result in results if result.name == "Risk Parity"
                )
                perfs.append(risk_parity)

            # Run the risk parity backtest for each risk mode
            for risk_mode in [
                ParityProcessorDelegate.RiskMode.LOW_RISK,
                ParityProcessorDelegate.RiskMode.MEDIUM_RISK,
                ParityProcessorDelegate.RiskMode.HIGH_RISK,
            ]:
                # print(f"Running risk parity for risk mode: {risk_mode.name}")
                parity_processor = ParityBacktestingProcessor(
                    perfs[0],
                    perfs[1],
                    perfs[2],
                    parity_lookback_period=parity_lookback_period,
                    mode=risk_mode,
                )
                parity_processor.delegate = ParityProcessorDelegate(risk_mode)
                parity_perf = parity_processor.backtest(initial_cash=initial_cash)
                all_parity_perfs.append(parity_perf)
                backtests[0].price_data = parity_processor.price_data()
                backtests[0].export_results(
                    performances=[parity_perf],
                    folder_path=f"./out/{rebalance_frequency}",
                    file_name=f"parity_{risk_mode.name}_{scenario}.xlsx",
                    export_csv=csv_export,
                )

                # Redo it with parity lookback period = None
                parity_processor = ParityBacktestingProcessor(
                    perfs[0].starting_from(
                        perfs[0].rebalance_dates[0]
                        + pd.Timedelta(days=parity_lookback_period)
                    ),
                    perfs[1].starting_from(
                        perfs[1].rebalance_dates[0]
                        + pd.Timedelta(days=parity_lookback_period)
                    ),
                    perfs[2].starting_from(
                        perfs[2].rebalance_dates[0]
                        + pd.Timedelta(days=parity_lookback_period)
                    ),
                    parity_lookback_period=None,
                    mode=risk_mode,
                )
                parity_processor.delegate = ParityProcessorDelegate(risk_mode)
                parity_perf = parity_processor.backtest(initial_cash=initial_cash)
                all_parity_perfs.append(parity_perf)
                backtests[0].price_data = parity_processor.price_data()
                backtests[0].export_results(
                    performances=[parity_perf],
                    folder_path=f"./out/{rebalance_frequency}",
                    file_name=f"parity_{risk_mode.name}_{scenario}_no_parity_lookback.xlsx",
                    export_csv=csv_export,
                )

    progress_logger.delete()
    return all_parity_perfs, all_asset_class_results


if __name__ == "__main__":
    rebalance_frequency = "1M"
    asset_class = None
    csv_export = False

    if "--rebalance" in sys.argv:
        rebalance_frequency = sys.argv[sys.argv.index("--rebalance") + 1]
    if "--class" in sys.argv:
        asset_class = sys.argv[sys.argv.index("--class") + 1]
    if "--csv" in sys.argv:
        csv_export = True

    parity_performances, asset_class_results = main(
        rebalance_frequency, asset_class, csv_export
    )
