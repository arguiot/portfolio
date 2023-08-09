from portfolio_optimization.data_processing import *
from portfolio_optimization.data_collection import *
from datetime import timedelta
from tokens.get_assets import *
import numpy as np
import copy
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


def run_for_asset_class(asset_list, asset_class="high_risk_tickers"):
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

    initial_bid = 1000

    porfolio_hrp = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=HRPOptimization,
        max_weight=max_weight,
    )

    portfolio_markowitz = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=Markowitz,
        max_weight=max_weight,
    )

    portfolio_bl = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=BlackLitterman,
        mcaps=mcaps.loc[start_date_portfolio],
        max_weight=max_weight,
    )

    portfolio_parity = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=RiskParity,
        max_weight=max_weight,
    )

    portfolio_fast_parity = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=FastRiskParity,
        max_weight=max_weight,
    )

    portfolio_default = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=Heuristic,
        max_weight=max_weight,
    )

    portfolio_rtr = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=RewardToRisk,
        max_weight=max_weight,
    )

    portfolio_vo = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=SimpleVolatility,
        max_weight=max_weight,
    )

    portfolio_vov = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=VolatilityOfVolatility,
        max_weight=max_weight,
    )

    portfolio_var = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=ValueAtRisk,
        max_weight=max_weight,
    )

    portfolio_rvar = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=RewardToVaR,
        max_weight=max_weight,
    )

    portfolio_combination = Portfolio(
        base_value=initial_bid,
        initial_prices=df.loc[:start_date_portfolio],
        optimiser=Combination,
        max_weight=max_weight,
    )

    backtest1d = Backtest(
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
        rebalance_frequency="1D",
        data=df,
        mcaps=mcaps,
    )

    # copy backtest1d to backtest1w
    backtest1w = copy.deepcopy(backtest1d)
    backtest1w.rebalance_frequency = "1W"

    # copy backtest1d to backtest1m
    backtest1m = copy.deepcopy(backtest1d)
    backtest1m.rebalance_frequency = "1M"

    yield_data = pd.Series()
    for asset in asset_list[asset_class]:
        if asset_class == "high_risk_tickers":
            yield_data[asset] = 0.075
        elif asset_class == "medium_risk_tickers":
            yield_data[asset] = 0
        elif asset_class == "low_risk_tickers":
            yield_data[asset] = 0.06
    perfs_1d = backtest1d.run_backtest(
        look_back_period=120, look_back_unit="D", yield_data=yield_data
    )
    perfs_1w = backtest1w.run_backtest(
        look_back_period=120, look_back_unit="D", yield_data=yield_data
    )
    perfs_1m = backtest1m.run_backtest(
        look_back_period=120, look_back_unit="D", yield_data=yield_data
    )

    backtest1d.export_results(
        perfs_1d, "./out/daily/", f"backtest_results_{asset_class}.xlsx"
    )
    backtest1w.export_results(
        perfs_1w, "./out/weekly/", f"backtest_results_{asset_class}.xlsx"
    )
    backtest1m.export_results(
        perfs_1m, "./out/monthly/", f"backtest_results_{asset_class}.xlsx"
    )


if __name__ == "__main__":
    asset_list = get_tickers()
    asset_classes = ["high_risk_tickers", "medium_risk_tickers", "low_risk_tickers"]
    for asset_class in asset_classes:
        run_for_asset_class(asset_list, asset_class=asset_class)
        # Print progress
        index = asset_classes.index(asset_class) + 1
        print(f"[PROGRESS]: {index * 100 / len(asset_classes)}%")
