from portfolio_optimization.data_processing import *
from portfolio_optimization.data_collection import *
from datetime import timedelta
from tokens.get_assets import *
import numpy as np
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
from main_backtest.delegates import (
    OptRebalancingPortfolioDelegate,
    CustomMarkowitzDelegate,
)

from portfolio_optimization.portfolio.Portfolio import Portfolio
from portfolio_optimization.backtesting.Backtesting import Backtest
from dateutil.relativedelta import relativedelta


def create_portfolios(
    asset_list,
    asset_class="high_risk_tickers",
    progress_logger=None,
    start_date=pd.to_datetime("2021-06-26"),
    end_date=pd.to_datetime("2024-06-25"),
    lookback_period=120,
    rebalance_frequency="1W",
    daily_trade_generator=True,
    initial_cash=1000000.0,
):
    _df = get_historical_prices_for_assets(
        asset_list[asset_class],
        interested_columns=["ReferenceRate", "CapMrktEstUSD"],
        start_date=start_date - pd.Timedelta(days=lookback_period),
        end_date=end_date,
    )

    print(f"[START DATE]: {start_date}")
    print(f"[DF Start Date]: {_df.index[0]}")

    # Filter out all columns containing `_` in their name
    df = _df.loc[:, ~_df.columns.str.contains("_")]

    # Get all the market caps
    mcaps = _df.loc[:, _df.columns.str.contains("CapMrktEstUSD")]
    mcaps.columns = mcaps.columns.str.replace("_CapMrktEstUSD", "")
    mcaps.replace(np.nan, 0, inplace=True)

    start_date_portfolio = start_date

    print(f"[START DATE]: {start_date_portfolio}")

    # Specify per asset as well
    max_weight = {"*": 1.0}  # "*" key must be higher than the rest.
    if asset_class == "high_risk_tickers":
        max_weight = {
            "*": 0.15,
            "avax": 0.05,
            "ftm": 0.05,
            "mkr": 0.05,
            "ondo": 0.05,
            "pendle": 0.05,
            "kas": 0.05,
        }
        # max_weight = {"*": 1.0}
    elif asset_class == "medium_risk_tickers":
        max_weight = {"*": 0.05}
        # max_weight = {"*": 1.0}
    elif asset_class == "low_risk_tickers":
        max_weight = {
            "classA": {
                "sum": 0.45,
                "assets": ["usdc", "usdt", "dai"],
            },
            "classB": {
                "sum": 0.45,
                "assets": [
                    "frax",
                    "gho",
                    "usdd",
                    "fdusd",
                    "usde",
                    "crvusd",
                    "susd",
                    "tusd",
                    "pyusd",
                ],
            },
            "volatile": {
                "sum": 0.10,
                "assets": [
                    "btc",
                    "eth",
                    "bnb",
                    "matic",
                    "xaut",
                    "paxg",
                ],
            },
            "usdc": 0.2,
            "usdt": 0.2,
            "dai": 0.2,
            "frax": 0.10,
            "gho": 0.10,
            "usdd": 0.10,
            "fdusd": 0.10,
            "usde": 0.10,
            "crvusd": 0.10,
            "susd": 0.10,
            "tusd": 0.10,
            "pyusd": 0.10,
            "btc": 0.10,
            "eth": 0.10,
            "bnb": 0.10,
            "matic": 0.10,
            "xaut": 0.10,
            "paxg": 0.10,
            "*": 1.0,
        }
        # Sum of cash = 90%
        # Sum of USDT, USDC, DAI = 60%
        # Rest of stable coins 30%
        # Maximum of gold = 5% each, sum = 10%

    min_weight = {"*": 0.0}
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

    initial_bid = initial_cash

    chosen_delegate = OptRebalancingPortfolioDelegate()
    # HeuristicRebalancingPortfolioDelegate()
    # OptRebalancingPortfolioDelegate()
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

    # portfolio_bl = Portfolio(
    #     base_value=initial_bid,
    #     initial_prices=df.loc[:start_date_portfolio],
    #     optimiser=BlackLitterman,
    #     mcaps=mcaps.loc[start_date_portfolio],
    #     max_weight=max_weight,
    #     min_weight=min_weight,
    # )

    # portfolio_bl.delegate = chosen_delegate

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

    # portfolio_fast_parity = Portfolio(
    #     base_value=initial_bid,
    #     initial_prices=df.loc[:start_date_portfolio],
    #     optimiser=FastRiskParity,
    #     max_weight=max_weight,
    #     min_weight=min_weight,
    # )

    # portfolio_fast_parity.delegate = chosen_delegate

    # portfolio_default = Portfolio(
    #     base_value=initial_bid,
    #     initial_prices=df.loc[:start_date_portfolio],
    #     optimiser=Heuristic,
    #     max_weight=max_weight,
    #     min_weight=min_weight,
    # )

    # portfolio_default.delegate = chosen_delegate

    # portfolio_rtr = Portfolio(
    #     base_value=initial_bid,
    #     initial_prices=df.loc[:start_date_portfolio],
    #     optimiser=RewardToRisk,
    #     max_weight=max_weight,
    #     min_weight=min_weight,
    # )

    # portfolio_rtr.delegate = chosen_delegate

    # portfolio_vo = Portfolio(
    #     base_value=initial_bid,
    #     initial_prices=df.loc[:start_date_portfolio],
    #     optimiser=SimpleVolatility,
    #     max_weight=max_weight,
    # )

    # portfolio_vo.delegate = chosen_delegate

    # # portfolio_vov = Portfolio(
    # #     base_value=initial_bid,
    # #     initial_prices=df.loc[:start_date_portfolio],
    # #     optimiser=VolatilityOfVolatility,
    # #     max_weight=max_weight,
    # #     min_weight=min_weight,
    # # )

    # # portfolio_vov.delegate = chosen_delegate

    # portfolio_var = Portfolio(
    #     base_value=initial_bid,
    #     initial_prices=df.loc[:start_date_portfolio],
    #     optimiser=ValueAtRisk,
    #     max_weight=max_weight,
    #     min_weight=min_weight,
    # )

    # portfolio_var.delegate = chosen_delegate

    # portfolio_rvar = Portfolio(
    #     base_value=initial_bid,
    #     initial_prices=df.loc[:start_date_portfolio],
    #     optimiser=RewardToVaR,
    #     max_weight=max_weight,
    #     min_weight=min_weight,
    # )

    # portfolio_rvar.delegate = chosen_delegate

    # portfolio_combination = Portfolio(
    #     base_value=initial_bid,
    #     initial_prices=df.loc[:start_date_portfolio],
    #     optimiser=Combination,
    #     max_weight=max_weight,
    #     min_weight=min_weight,
    # )

    # portfolio_combination.delegate = chosen_delegate

    print(f"[REBALANCE FREQUENCY]: {rebalance_frequency}")

    portfolios = {
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
    }

    if asset_class != "low_risk_tickers":
        portfolios["HRP"] = porfolio_hrp

    backtest = Backtest(
        portfolios=portfolios,
        start_date=start_date_portfolio,
        end_date=df.index[-1],
        rebalance_frequency=rebalance_frequency,
        adjust_holdings=daily_trade_generator,
        data=df,
        mcaps=mcaps,
        asset_class=asset_class,
        progress_logger=progress_logger,
    )

    return backtest
