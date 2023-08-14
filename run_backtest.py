from portfolio_optimization.data_processing import *
from portfolio_optimization.data_collection import *
from datetime import timedelta
from tokens.get_assets import *
import numpy as np
import copy
import sys
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

from portfolio_optimization.portfolio.Portfolio import Portfolio
from portfolio_optimization.backtesting.Backtesting import Backtest
from dateutil.relativedelta import relativedelta


def run_for_asset_class(
    asset_list,
    asset_class="high_risk_tickers",
    show_progress=False,
    rebalance_frequency="1D",
):
    _df = get_historical_prices_for_assets(
        asset_list[asset_class],
        time_range=timedelta(days=365 * 3),
        interested_columns=["ReferenceRate", "CapMrktEstUSD"],
    )

    # Filter out all columns containing `_` in their name
    df = _df.loc[:, ~_df.columns.str.contains("_")]

    # Get all the market caps
    mcaps = _df.loc[:, _df.columns.str.contains("CapMrktEstUSD")]
    mcaps.columns = mcaps.columns.str.replace("_CapMrktEstUSD", "")
    mcaps.replace(np.nan, 0, inplace=True)

    start_date_portfolio = df.index[0] + relativedelta(months=5)

    max_weight = 1
    if asset_class == "high_risk_tickers":
        max_weight = 0.15
    elif asset_class == "medium_risk_tickers":
        max_weight = 0.05
    elif asset_class == "low_risk_tickers":
        max_weight = 0.3

    weight_threshold = 0.01

    if asset_class == "high_risk_tickers":
        weight_threshold = 0.05
    elif asset_class == "medium_risk_tickers":
        weight_threshold = 0.03
    elif asset_class == "low_risk_tickers":
        weight_threshold = 0.01

    initial_bid = 1000

    porfolio_hrp = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=HRPOptimization,
        max_weight=max_weight,
        weight_threshold=weight_threshold,
    )

    portfolio_markowitz = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=Markowitz,
        max_weight=max_weight,
        weight_threshold=weight_threshold,
    )

    portfolio_bl = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=BlackLitterman,
        mcaps=mcaps.loc[start_date_portfolio],
        max_weight=max_weight,
        weight_threshold=weight_threshold,
    )

    portfolio_parity = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=RiskParity,
        max_weight=max_weight,
        weight_threshold=weight_threshold,
    )

    portfolio_fast_parity = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=FastRiskParity,
        max_weight=max_weight,
        weight_threshold=weight_threshold,
    )

    portfolio_default = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=Heuristic,
        max_weight=max_weight,
        weight_threshold=weight_threshold,
    )

    portfolio_rtr = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=RewardToRisk,
        max_weight=max_weight,
        weight_threshold=weight_threshold,
    )

    portfolio_vo = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=SimpleVolatility,
        max_weight=max_weight,
        weight_threshold=weight_threshold,
    )

    portfolio_vov = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=VolatilityOfVolatility,
        max_weight=max_weight,
        weight_threshold=weight_threshold,
    )

    portfolio_var = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=ValueAtRisk,
        max_weight=max_weight,
        weight_threshold=weight_threshold,
    )

    portfolio_rvar = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=RewardToVaR,
        max_weight=max_weight,
        weight_threshold=weight_threshold,
    )

    portfolio_combination = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=Combination,
        max_weight=max_weight,
        weight_threshold=weight_threshold,
    )

    print(f"[REBALANCE FREQUENCY]: {rebalance_frequency}")

    backtest = Backtest(
        portfolios={
            "HRP": porfolio_hrp,
            "Markowitz": portfolio_markowitz,
            "BL": portfolio_bl,
            "Risk Parity": portfolio_parity,
            "Fast Risk Parity": portfolio_fast_parity,
            "Heuristic": portfolio_default,
            "Reward to Risk": portfolio_rtr,
            "Simple Volatility": portfolio_vo,
            "Volatility of Volatility": portfolio_vov,
            "Value At Risk": portfolio_var,
            "Reward to VaR": portfolio_rvar,
            "Combination": portfolio_combination,
        },
        start_date=start_date_portfolio,
        end_date=df.index[-1],
        rebalance_frequency=rebalance_frequency,
        data=df,
        mcaps=mcaps,
    )

    yield_data = pd.Series()
    for asset in asset_list[asset_class]:
        if asset_class == "high_risk_tickers":
            yield_data[asset] = 0.075
        elif asset_class == "medium_risk_tickers":
            yield_data[asset] = 0
        elif asset_class == "low_risk_tickers":
            yield_data[asset] = 0.06

    if show_progress:
        print(f"[PROGRESS]: 10%")

    perfs = backtest.run_backtest(
        look_back_period=120, look_back_unit="D", yield_data=yield_data
    )
    if show_progress:
        print(f"[PROGRESS]: 90%")

    # Check if the directory exists, if not, create it
    if not os.path.exists(f"./out/{rebalance_frequency}/"):
        os.makedirs(f"./out/{rebalance_frequency}/")

    backtest.export_results(
        perfs, f"./out/{rebalance_frequency}/", f"backtest_results_{asset_class}.xlsx"
    )

    if show_progress:
        print(f"[PROGRESS]: 100%")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    asset_list = get_tickers()
    # Get `--rebalance <frequency>` flag
    if "--rebalance" in sys.argv:
        rebalance_frequency = sys.argv[sys.argv.index("--rebalance") + 1]
    else:
        rebalance_frequency = "1M"
    # If the CLI is called with the `--class <asset_class>` flag, run the backtest for the specified asset class
    if "--class" in sys.argv:
        asset_class = sys.argv[sys.argv.index("--class") + 1]
        run_for_asset_class(
            asset_list,
            asset_class=asset_class,
            show_progress=True,
            rebalance_frequency=rebalance_frequency,
        )
        sys.exit()
    else:
        # Otherwise, run the backtest for all asset classes
        asset_classes = ["high_risk_tickers", "medium_risk_tickers", "low_risk_tickers"]
        for asset_class in asset_classes:
            run_for_asset_class(
                asset_list,
                asset_class=asset_class,
                show_progress=True,
                rebalance_frequency=rebalance_frequency,
            )
            # Print progress
            index = asset_classes.index(asset_class) + 1
            print(f"[PROGRESS]: {index * 100 / len(asset_classes)}%")
