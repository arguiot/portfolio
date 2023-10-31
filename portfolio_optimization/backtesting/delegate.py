from portfolio_optimization.portfolio import Portfolio
from datetime import datetime as DateTime


class BacktestingDelegate:
    def post_process(self, backtest, portfolio: Portfolio, date: DateTime):
        pass
