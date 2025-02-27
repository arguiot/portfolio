import pandas as pd
import numpy as np
from portfolio_optimization.data_processing import *
from portfolio_optimization.data_collection import *
from tokens.get_assets import *
import sys
import os
from portfolio_optimization.utils import ProgressLogger

from portfolio_optimization.backtesting.Backtesting import (
    Backtest,
    PortfolioPerformance,
)
from portfolio_optimization.backtesting.parity import (
    ParityLine,
    ParityBacktestingProcessor,
    ParityProcessorDelegate,
    BTCParityProcessorDelegate,
    SOLParityProcessorDelegate,
    MATICParityProcessorDelegate,
    BNBParityProcessorDelegate,
    XRPParityProcessorDelegate,
    XLMParityProcessorDelegate,
    SUIParityProcessorDelegate,
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
        "end_date": pd.to_datetime("2025-01-01"),  # pd.to_datetime("2024-12-17"),
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


def create_portfolio_performance(asset, start_date, end_date):
    asset_data = get_historical_prices_for_assets(
        [asset],
        interested_columns=["ReferenceRate", "CapMrktEstUSD"],
        start_date=start_date - pd.Timedelta(days=lookback_period),
        end_date=end_date,
    )

    portfolio_value = pd.DataFrame({f"Portfolio Value": asset_data[asset]})
    rebalance_dates = portfolio_value.index
    portfolio_compositions = pd.Series(1, index=rebalance_dates, name=asset)
    portfolio_raw_composition = portfolio_compositions.copy()
    portfolio_holdings = pd.Series(
        1 / asset_data[asset], index=rebalance_dates, name=asset
    )
    portfolio_live_weights = pd.Series(1, index=rebalance_dates, name=asset)

    return PortfolioPerformance(
        portfolio_name=asset,
        portfolio_value=portfolio_value,
        rebalance_dates=rebalance_dates,  # type: ignore
        portfolio_compositions=portfolio_compositions,
        portfolio_raw_composition=portfolio_raw_composition,
        portfolio_holdings=portfolio_holdings,
        portfolio_live_weights=portfolio_live_weights,
    )


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
    all_btc_parity_perfs: List[PortfolioPerformance] = []
    all_sol_parity_perfs: List[PortfolioPerformance] = []
    all_matic_parity_perfs: List[PortfolioPerformance] = []
    all_bnb_parity_perfs: List[PortfolioPerformance] = []

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

            # Create Bitcoin portfolio performance
            btc_perf = create_portfolio_performance("btc", start_date, end_date)
            sol_perf = create_portfolio_performance("sol", start_date, end_date)
            matic_perf = create_portfolio_performance("matic", start_date, end_date)
            bnb_perf = create_portfolio_performance("bnb", start_date, end_date)
            xrp_perf = create_portfolio_performance("xrp", start_date, end_date)
            xlm_perf = create_portfolio_performance("xlm", start_date, end_date)
            sui_perf = create_portfolio_performance("sui", start_date, end_date)

            # Run the risk parity backtest for each risk mode
            for risk_mode in [
                ParityProcessorDelegate.RiskMode.LOW_RISK,
                ParityProcessorDelegate.RiskMode.MEDIUM_RISK,
                ParityProcessorDelegate.RiskMode.HIGH_RISK,
            ]:
                # Regular parity backtest
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

                # Bitcoin parity backtest
                btc_delegate = BTCParityProcessorDelegate(risk_mode)
                btc_parity_processor = ParityBacktestingProcessor(
                    btc_perf,
                    None,
                    perfs[2],
                    parity_lookback_period=parity_lookback_period,
                    delegate=btc_delegate,
                    mode=risk_mode,
                )
                btc_parity_perf = btc_parity_processor.backtest(
                    initial_cash=initial_cash
                )
                all_btc_parity_perfs.append(btc_parity_perf)
                backtests[0].price_data = btc_parity_processor.price_data()
                backtests[0].export_results(
                    performances=[btc_parity_perf],
                    folder_path=f"./out/{rebalance_frequency}",
                    file_name=f"btc_parity_{risk_mode.name}_{scenario}.xlsx",
                    export_csv=csv_export,
                )

                # Sol parity backtest
                sol_delegate = SOLParityProcessorDelegate(risk_mode)
                sol_parity_processor = ParityBacktestingProcessor(
                    sol_perf,
                    None,
                    perfs[2],
                    parity_lookback_period=parity_lookback_period,
                    delegate=sol_delegate,
                    mode=risk_mode,
                )
                sol_parity_perf = sol_parity_processor.backtest(
                    initial_cash=initial_cash
                )
                all_sol_parity_perfs.append(sol_parity_perf)
                backtests[0].price_data = sol_parity_processor.price_data()
                backtests[0].export_results(
                    performances=[sol_parity_perf],
                    folder_path=f"./out/{rebalance_frequency}",
                    file_name=f"sol_parity_{risk_mode.name}_{scenario}.xlsx",
                    export_csv=csv_export,
                )

                # MATIC parity backtest
                matic_delegate = MATICParityProcessorDelegate(risk_mode)
                matic_parity_processor = ParityBacktestingProcessor(
                    matic_perf,
                    None,
                    perfs[2],
                    parity_lookback_period=parity_lookback_period,
                    delegate=matic_delegate,
                    mode=risk_mode,
                )
                matic_parity_perf = matic_parity_processor.backtest(
                    initial_cash=initial_cash
                )
                all_matic_parity_perfs.append(matic_parity_perf)
                backtests[0].price_data = matic_parity_processor.price_data()
                backtests[0].export_results(
                    performances=[matic_parity_perf],
                    folder_path=f"./out/{rebalance_frequency}",
                    file_name=f"matic_parity_{risk_mode.name}_{scenario}.xlsx",
                    export_csv=csv_export,
                )

                # BNB parity backtest
                bnb_delegate = BNBParityProcessorDelegate(risk_mode)
                bnb_parity_processor = ParityBacktestingProcessor(
                    bnb_perf,
                    None,
                    perfs[2],
                    parity_lookback_period=parity_lookback_period,
                    delegate=bnb_delegate,
                    mode=risk_mode,
                )
                bnb_parity_perf = bnb_parity_processor.backtest(
                    initial_cash=initial_cash
                )
                all_bnb_parity_perfs.append(bnb_parity_perf)
                backtests[0].price_data = bnb_parity_processor.price_data()
                backtests[0].export_results(
                    performances=[bnb_parity_perf],
                    folder_path=f"./out/{rebalance_frequency}",
                    file_name=f"bnb_parity_{risk_mode.name}_{scenario}.xlsx",
                    export_csv=csv_export,
                )

                # Redo regular parity with parity lookback period = None
                parity_delegate = ParityProcessorDelegate(risk_mode)
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
                    delegate=parity_delegate,
                    mode=risk_mode,
                )
                parity_processor.delegate = parity_delegate
                parity_perf = parity_processor.backtest(initial_cash=initial_cash)
                all_parity_perfs.append(parity_perf)
                backtests[0].price_data = parity_processor.price_data()
                backtests[0].export_results(
                    performances=[parity_perf],
                    folder_path=f"./out/{rebalance_frequency}",
                    file_name=f"parity_{risk_mode.name}_{scenario}_no_parity_lookback.xlsx",
                    export_csv=csv_export,
                )

                # Do XRP parity backtest
                xrp_delegate = XRPParityProcessorDelegate(risk_mode)
                xrp_parity_processor = ParityBacktestingProcessor(
                    xrp_perf,
                    None,
                    perfs[2],
                    parity_lookback_period=parity_lookback_period,
                    delegate=xrp_delegate,
                    mode=risk_mode,
                )
                xrp_parity_perf = xrp_parity_processor.backtest(
                    initial_cash=initial_cash
                )
                all_parity_perfs.append(xrp_parity_perf)
                backtests[0].price_data = xrp_parity_processor.price_data()
                backtests[0].export_results(
                    performances=[xrp_parity_perf],
                    folder_path=f"./out/{rebalance_frequency}",
                    file_name=f"xrp_parity_{risk_mode.name}_{scenario}.xlsx",
                    export_csv=csv_export,
                )

                # Do XLM parity backtest
                xlm_delegate = XLMParityProcessorDelegate(risk_mode)
                xlm_parity_processor = ParityBacktestingProcessor(
                    xlm_perf,
                    None,
                    perfs[2],
                    parity_lookback_period=parity_lookback_period,
                    delegate=xlm_delegate,
                    mode=risk_mode,
                )
                xlm_parity_perf = xlm_parity_processor.backtest(
                    initial_cash=initial_cash
                )
                all_parity_perfs.append(xlm_parity_perf)
                backtests[0].price_data = xlm_parity_processor.price_data()
                backtests[0].export_results(
                    performances=[xlm_parity_perf],
                    folder_path=f"./out/{rebalance_frequency}",
                    file_name=f"xlm_parity_{risk_mode.name}_{scenario}.xlsx",
                    export_csv=csv_export,
                )

                # Do SUI parity backtest
                sui_delegate = SUIParityProcessorDelegate(risk_mode)
                sui_parity_processor = ParityBacktestingProcessor(
                    sui_perf,
                    None,
                    perfs[2],
                    parity_lookback_period=parity_lookback_period,
                    delegate=sui_delegate,
                    mode=risk_mode,
                )
                sui_parity_perf = sui_parity_processor.backtest(
                    initial_cash=initial_cash
                )
                all_parity_perfs.append(sui_parity_perf)
                backtests[0].price_data = sui_parity_processor.price_data()
                backtests[0].export_results(
                    performances=[sui_parity_perf],
                    folder_path=f"./out/{rebalance_frequency}",
                    file_name=f"sui_parity_{risk_mode.name}_{scenario}.xlsx",
                    export_csv=csv_export,
                )

    progress_logger.delete()
    return all_parity_perfs, all_asset_class_results, all_btc_parity_perfs


if __name__ == "__main__":
    rebalance_frequency = "1W"
    asset_class = None
    csv_export = False

    if "--rebalance" in sys.argv:
        rebalance_frequency = sys.argv[sys.argv.index("--rebalance") + 1]
    if "--class" in sys.argv:
        asset_class = sys.argv[sys.argv.index("--class") + 1]
    if "--csv" in sys.argv:
        csv_export = True

    parity_performances, asset_class_results, btc_parity_performances = main(
        rebalance_frequency, asset_class, csv_export
    )
