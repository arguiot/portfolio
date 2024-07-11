import pandas as pd
from portfolio_optimization.data_processing import *
from portfolio_optimization.data_collection import *
from tokens.get_assets import *
import sys
from portfolio_optimization.utils import ProgressLogger

from portfolio_optimization.backtesting.Backtesting import Backtest
from portfolio_optimization.backtesting.parity import (
    ParityLine,
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

daily_trade_generator = True


def run_for_asset_class(
    backtest: Backtest,
    rebalance_frequency="1D",
    progress_logger=None,
    scenario_tag=None,
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

    backtest.export_results(perfs, f"./out/{rebalance_frequency}/", filename)

    return perfs


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    progress_logger = ProgressLogger()
    asset_list = get_tickers()

    if "--rebalance" in sys.argv:
        rebalance_frequency = sys.argv[sys.argv.index("--rebalance") + 1]
    else:
        rebalance_frequency = "1M"

    for scenario, dates in scenarios.items():
        print(f"Running scenario: {scenario}")
        start_date = dates["start_date"]
        end_date = dates["end_date"]

        if "--class" in sys.argv:
            asset_class = sys.argv[sys.argv.index("--class") + 1]
            backtest = create_portfolios(
                asset_list,
                asset_class=asset_class,
                progress_logger=progress_logger.context(f"{scenario}_{asset_class}"),
                start_date=start_date,
                end_date=end_date,
                lookback_period=lookback_period,
                rebalance_frequency=rebalance_frequency,
                daily_trade_generator=daily_trade_generator,
            )
            run_for_asset_class(
                backtest=backtest,
                rebalance_frequency=rebalance_frequency,
                progress_logger=progress_logger,
                scenario_tag=scenario,
            )
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
                )
                risk_parity = next(
                    result for result in results if result.name == "Risk Parity"
                )
                perfs.append(risk_parity)

            # parity_line = ParityLine()
            # parity_line.regression(perfs[0], perfs[1], perfs[2])

            # print(f"[PARITY LINE for {scenario}]:")
            # print(f"Sigma A: {parity_line.sigma_a}")
            # print(f"Sigma B: {parity_line.sigma_b}")
            # print(f"Sigma C: {parity_line.sigma_c}")
            # print(f"Return A: {parity_line.r_a}")
            # print(f"Return B: {parity_line.r_b}")
            # print(f"Return C: {parity_line.r_c}")
            # print(f"Weight A: {parity_line.weight_A}")
            # print(f"Weight B: {parity_line.weight_B}")

    # parity_processor = ParityBacktestingProcessor(
    #     perfs[0],
    #     perfs[1],
    #     perfs[2],
    # )
    # parity_processor.delegate = CustomParityDelegate()
    # parity_perf = parity_processor.backtest(initial_cash=3000)
    # backtests[0].price_data = parity_processor.price_data()
    # backtests[0].export_results(
    #     performances=[parity_perf],
    #     folder_path=f"./out/{rebalance_frequency}",
    #     file_name="parity.xlsx",
    # )
    progress_logger.delete()
