import pandas as pd
from portfolio_optimization.data_processing import *
from portfolio_optimization.data_collection import *
from datetime import timedelta
from tokens.get_assets import *
import numpy as np
from pypfopt import expected_returns
import sys
from portfolio_optimization.optimization.GeneralOptimization import (
    GeneralOptimizationDelegate,
)
from portfolio_optimization.optimization.hrp import HRPOptimization
from portfolio_optimization.optimization.markowitz import Markowitz
from portfolio_optimization.optimization.black_litterman import BlackLitterman
from portfolio_optimization.optimization.risk_parity import RiskParity
from portfolio_optimization.optimization.heuristic import (
    FastRiskParity,
    Heuristic,
    RewardToRisk,
    SimpleVolatility,
    VolatilityOfVolatility,
    ValueAtRisk,
    RewardToVaR,
    Combination,
)
from portfolio_optimization.utils import ProgressLogger

from portfolio_optimization.portfolio.Portfolio import Portfolio
from portfolio_optimization.portfolio.delegate import PortfolioDelegate
from portfolio_optimization.portfolio.rebalancing import (
    optimize_trades,
    deterministic_optimal_rebalancing,
)
from portfolio_optimization.backtesting.parity import ParityBacktestingProcessor
from portfolio_optimization.backtesting.Backtesting import Backtest
from portfolio_optimization.backtesting.parity import (
    ParityProcessorDelegate,
    ParityLine,
)
from dateutil.relativedelta import relativedelta
from portfolio_optimization.portfolio.trade_generator import totalOrders, orderSize


class OptRebalancingPortfolioDelegate(PortfolioDelegate):
    def rebalance(
        self,
        holdings: pd.Series,
        prices: pd.Series,
        target_weights: pd.Series,
        base_value,
    ) -> pd.Series:
        diff = optimize_trades(  # or deterministic_optimal_rebalancing
            base_value=base_value,
            holdings=holdings,
            new_target_weights=target_weights,
            prices=prices,
            min_W=target_weights * 0.9,
            max_W=target_weights * 1.1,
            external_movement=0,
        )

        new_holdings = diff + holdings
        return new_holdings


class HeuristicRebalancingPortfolioDelegate(PortfolioDelegate):
    def rebalance(
        self,
        holdings: pd.Series,
        prices: pd.Series,
        target_weights: pd.Series,
        base_value,
    ) -> pd.Series:
        tbd = 0
        weight_assets = [
            (
                target_weights[i]
                if i in target_weights
                else 0,  # - 0.1 * target_weights[i],
                target_weights[i] if i in target_weights else 0,
                target_weights[i]
                if i in target_weights
                else 0,  # + 0.1 * target_weights[i]
            )
            for i in prices.index
        ]
        order_size_assets = [(50, 1000) for i in prices.index]

        _holdings = holdings.reindex(prices.index).fillna(0)
        _prices = prices.reindex(prices.index).fillna(0)

        orders = totalOrders(
            np.array(_holdings),
            np.array(_prices),
            np.array(order_size_assets),
            np.array(weight_assets),
            tbd,
        )

        sizes = orderSize(
            np.array(_holdings),
            np.array(_prices),
            np.array(order_size_assets),
            np.array(weight_assets),
            tbd,
        )

        trades = orders * sizes

        return _holdings + trades / prices


class CustomMarkowitzDelegate(GeneralOptimizationDelegate):
    def setup(self, optimization_object: Markowitz):
        optimization_object.mode = optimization_object.CovMode.LEDOIT_WOLF
        optimization_object.efficient_portfolio = (
            optimization_object.EfficientPortfolio.MAX_SHARPE
        )

        if optimization_object.cov_matrix is None:
            optimization_object.cov_matrix = optimization_object.get_cov_matrix()

        optimization_object.rets = expected_returns.mean_historical_return(
            optimization_object.df
        )

        return super().setup(optimization_object)


class CustomParityDelegate(ParityProcessorDelegate):
    def compute_weights(self, parity_line: ParityLine) -> pd.Series:
        _return = parity_line.getMaxReturn()
        print(f"Max Return: {_return}")
        weights = parity_line.convertReturn(_return)[1:]
        return pd.Series(weights)


def create_portfolios(
    asset_list,
    asset_class="high_risk_tickers",
    progress_logger=None,
):
    _df = get_historical_prices_for_assets(
        asset_list[asset_class],
        time_range=timedelta(days=365 * 3),  # 3 years
        interested_columns=["ReferenceRate", "CapMrktEstUSD"],
    )

    # Filter out all columns containing `_` in their name
    df = _df.loc[:, ~_df.columns.str.contains("_")]

    # Get all the market caps
    mcaps = _df.loc[:, _df.columns.str.contains("CapMrktEstUSD")]
    mcaps.columns = mcaps.columns.str.replace("_CapMrktEstUSD", "")
    mcaps.replace(np.nan, 0, inplace=True)

    start_date_portfolio = df.index[0] + relativedelta(days=120)

    # Specify per asset as well
    max_weight = {"*": 1.0}
    # if asset_class == "high_risk_tickers":
    #     max_weight = {"*": 0.15}
    # elif asset_class == "medium_risk_tickers":
    #     max_weight = 0.05
    # elif asset_class == "low_risk_tickers":
    #     max_weight = 0.25

    min_weight = {"*": 0}
    # min_weight = { 'btc': 0.05, '*': 0.01 }
    # if asset_class == "high_risk_tickers":
    #     min_weight = {"*": 0.01}
    # elif asset_class == "medium_risk_tickers":
    #     min_weight = 0.01
    # elif asset_class == "low_risk_tickers":
    #     min_weight = 0.01

    budget = {}
    if asset_class == "high_risk_tickers":
        budget = {}
    elif asset_class == "medium_risk_tickers":
        budget = {}
    elif asset_class == "low_risk_tickers":
        budget = {}

    initial_bid = 1000.0

    chosen_delegate = PortfolioDelegate()
    # (
    #     OptRebalancingPortfolioDelegate()
    # )  # Or HeuristicRebalancingPortfolioDelegate()

    porfolio_hrp = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=HRPOptimization,
        max_weight=max_weight,
        min_weight=min_weight,
    )

    porfolio_hrp.delegate = chosen_delegate

    portfolio_markowitz = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=Markowitz.bind(CustomMarkowitzDelegate()),
        max_weight=max_weight,
        min_weight=min_weight,
    )

    portfolio_markowitz.delegate = chosen_delegate

    portfolio_bl = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=BlackLitterman,
        mcaps=mcaps.loc[start_date_portfolio],
        max_weight=max_weight,
        min_weight=min_weight,
    )

    portfolio_bl.delegate = chosen_delegate

    portfolio_parity = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=RiskParity,
        max_weight=max_weight,
        min_weight=min_weight,
        budget=budget,
        lambda_var=0.1,
        lambda_u=0.1,
    )

    portfolio_parity.delegate = chosen_delegate

    portfolio_fast_parity = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=FastRiskParity,
        max_weight=max_weight,
        min_weight=min_weight,
    )

    portfolio_fast_parity.delegate = chosen_delegate

    portfolio_default = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=Heuristic,
        max_weight=max_weight,
        min_weight=min_weight,
    )

    portfolio_default.delegate = chosen_delegate

    portfolio_rtr = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=RewardToRisk,
        max_weight=max_weight,
        min_weight=min_weight,
    )

    portfolio_rtr.delegate = chosen_delegate

    portfolio_vo = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=SimpleVolatility,
        max_weight=max_weight,
    )

    portfolio_vo.delegate = chosen_delegate

    portfolio_vov = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=VolatilityOfVolatility,
        max_weight=max_weight,
        min_weight=min_weight,
    )

    portfolio_vov.delegate = chosen_delegate

    portfolio_var = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=ValueAtRisk,
        max_weight=max_weight,
        min_weight=min_weight,
    )

    portfolio_var.delegate = chosen_delegate

    portfolio_rvar = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=RewardToVaR,
        max_weight=max_weight,
        min_weight=min_weight,
    )

    portfolio_rvar.delegate = chosen_delegate

    portfolio_combination = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=Combination,
        max_weight=max_weight,
        min_weight=min_weight,
    )

    portfolio_combination.delegate = chosen_delegate

    print(f"[REBALANCE FREQUENCY]: {rebalance_frequency}")

    backtest = Backtest(
        portfolios={
            "HRP": porfolio_hrp,
            "Markowitz": portfolio_markowitz,
            # "BL": portfolio_bl,
            "Risk Parity": portfolio_parity,
            # "Fast Risk Parity": portfolio_fast_parity,
            # "Heuristic": portfolio_default,
            # "Reward to Risk": portfolio_rtr,
            # "Simple Volatility": portfolio_vo,
            # "Volatility of Volatility": portfolio_vov,
            # "Value At Risk": portfolio_var,
            # "Reward to VaR": portfolio_rvar,
            # "Combination": portfolio_combination,
        },
        start_date=start_date_portfolio,
        end_date=df.index[-1],
        rebalance_frequency=rebalance_frequency,
        adjust_holdings=False,
        data=df,
        mcaps=mcaps,
        asset_class=asset_class,
        progress_logger=progress_logger,
    )

    return backtest


def run_for_asset_class(
    backtest: Backtest,
    rebalance_frequency="1D",
    progress_logger=None,
):
    yield_data = pd.Series()
    for asset in asset_list[asset_class]:
        if asset_class == "high_risk_tickers":
            yield_data[asset] = 0.075
        elif asset_class == "medium_risk_tickers":
            yield_data[asset] = 0
        elif asset_class == "low_risk_tickers":
            yield_data[asset] = 0.06

    perfs = backtest.run_backtest(
        look_back_period=120,
        look_back_unit="D",
        yield_data=yield_data,
        progress_logger=progress_logger,
    )

    # Check if the directory exists, if not, create it
    if not os.path.exists(f"./out/{rebalance_frequency}/"):
        os.makedirs(f"./out/{rebalance_frequency}/")

    backtest.export_results(
        perfs, f"./out/{rebalance_frequency}/", f"backtest_results_{asset_class}.xlsx"
    )

    return perfs


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    # Instantiate the `ProgressLogger` object
    progress_logger = ProgressLogger()

    asset_list = get_tickers()
    # Get `--rebalance <frequency>` flag
    if "--rebalance" in sys.argv:
        rebalance_frequency = sys.argv[sys.argv.index("--rebalance") + 1]
    else:
        rebalance_frequency = "1M"
    # If the CLI is called with the `--class <asset_class>` flag, run the backtest for the specified asset class
    if "--class" in sys.argv:
        asset_class = sys.argv[sys.argv.index("--class") + 1]
        backtest = create_portfolios(
            asset_list,
            asset_class=asset_class,
            progress_logger=progress_logger.context(asset_class),
        )
        run_for_asset_class(
            backtest=backtest,
            rebalance_frequency=rebalance_frequency,
            progress_logger=progress_logger,
        )
        sys.exit()
    else:
        # Otherwise, run the backtest for all asset classes
        asset_classes = ["high_risk_tickers", "medium_risk_tickers", "low_risk_tickers"]

        backtests = []
        loggers = []

        for asset_class in asset_classes:
            logger = progress_logger.context(asset_class)

            backtest = create_portfolios(
                asset_list,
                asset_class=asset_class,
                progress_logger=logger,
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
            )
            # Get the 'Risk Parity' portfolio, so first where .name == 'Risk Parity'
            risk_parity = next(
                result for result in results if result.name == "Risk Parity"
            )
            perfs.append(risk_parity)

        parity_processor = ParityBacktestingProcessor(
            perfs[0],
            perfs[1],
            perfs[2],
        )
        parity_processor.delegate = CustomParityDelegate()
        parity_perf = parity_processor.backtest(initial_cash=1000)
        backtests[0].price_data = parity_processor.price_data()
        backtests[0].export_results(
            performances=[parity_perf],
            folder_path=f"./out/{rebalance_frequency}",
            file_name="parity.xlsx",
        )
    progress_logger.delete()
