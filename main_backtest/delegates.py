import pandas as pd
from portfolio_optimization.data_processing import *
from portfolio_optimization.data_collection import *
from tokens.get_assets import *
import numpy as np
from pypfopt import expected_returns
from portfolio_optimization.optimization.GeneralOptimization import (
    GeneralOptimizationDelegate,
)
from portfolio_optimization.optimization.markowitz import Markowitz

from portfolio_optimization.portfolio.delegate import PortfolioDelegate
from portfolio_optimization.portfolio.rebalancing import (
    optimize_trades,
)
from portfolio_optimization.backtesting.parity import (
    ParityProcessorDelegate,
    ParityLine,
)
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
                (
                    target_weights[i] if i in target_weights else 0
                ),  # - 0.1 * target_weights[i],
                target_weights[i] if i in target_weights else 0,
                (
                    target_weights[i] if i in target_weights else 0
                ),  # + 0.1 * target_weights[i]
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
