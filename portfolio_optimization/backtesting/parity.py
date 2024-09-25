from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
from portfolio_optimization.backtesting import Backtest
from portfolio_optimization.portfolio import Portfolio
from .Backtesting import PortfolioPerformance
from ..optimization.risk_parity import RiskParity
from .delegate import BacktestingDelegate
from enum import Enum


class ParityLine:
    def __init__(self):
        self.weight_A = 0.8
        self.weight_B = 0.2
        self.sigma_a = 0.0
        self.sigma_b = 0.0
        self.sigma_c = 0.0
        self.sigma_ab = 0.0
        self.r_a = 0.0
        self.r_b = 0.0
        self.r_c = 0.0

    def regression(
        self,
        portfolio_a: PortfolioPerformance,
        portfolio_b: PortfolioPerformance,
        portfolio_g: PortfolioPerformance,
    ):
        # Combine a and b
        # Compute weight A and B, using risk parity
        rp = RiskParity(
            df=pd.concat(
                [
                    portfolio_a.portfolio_value["Portfolio Value"],
                    portfolio_b.portfolio_value["Portfolio Value"],
                ],
                axis=1,
            )
        )
        _weights = rp.get_weights()
        self.weight_A = _weights.iloc[0]
        self.weight_B = _weights.iloc[1]
        # Calculate the average annualized return (APY)
        start_date_a = portfolio_a.portfolio_value["Portfolio Value"].index[0]
        end_date_a = portfolio_a.portfolio_value["Portfolio Value"].index[-1]
        years_a = (end_date_a - start_date_a).days / 365
        self.r_a = (
            (
                portfolio_a.portfolio_value["Portfolio Value"].iloc[-1]
                / portfolio_a.portfolio_value["Portfolio Value"].iloc[0]
            )
            ** (1 / years_a)
        ) - 1

        start_date_b = portfolio_b.portfolio_value["Portfolio Value"].index[0]
        end_date_b = portfolio_b.portfolio_value["Portfolio Value"].index[-1]
        years_b = (end_date_b - start_date_b).days / 365
        self.r_b = (
            (
                portfolio_b.portfolio_value["Portfolio Value"].iloc[-1]
                / portfolio_b.portfolio_value["Portfolio Value"].iloc[0]
            )
            ** (1 / years_b)
        ) - 1
        self.sigma_a = portfolio_a.portfolio_value[
            "Portfolio Value"
        ].pct_change().std() * np.sqrt(365)
        self.sigma_b = portfolio_b.portfolio_value[
            "Portfolio Value"
        ].pct_change().std() * np.sqrt(365)

        # Add the two portfolio values (weights * value) to compute sigma_ab
        portfolio_ab = (
            portfolio_a.portfolio_value["Portfolio Value"] * self.weight_A
            + portfolio_b.portfolio_value["Portfolio Value"] * self.weight_B
        )
        self.sigma_ab = portfolio_ab.pct_change().std() * np.sqrt(365)
        self.sigma_c = portfolio_g.portfolio_value[
            "Portfolio Value"
        ].pct_change().std() * np.sqrt(365)

    def getMinRisk(self):
        return 0

    def getMaxRisk(self):
        return 0.8

    def convertReturn(self, _return):
        return_riskless = 0.1
        return_risky = 0.5
        sigma_riskless = self.sigma_c
        sigma_risky = self.sigma_ab

        target_sigma = min(
            max(
                sigma_riskless
                + ((_return - return_riskless) / (return_risky - return_riskless))
                * (sigma_risky - sigma_riskless),
                sigma_riskless,
            ),
            0.8,  # 80%
        )

        return target_sigma

    def convertWeights(self, weight_alpha, weight_beta, weight_gamma):
        _return = (
            weight_alpha * self.r_a + weight_beta * self.r_b + weight_gamma * self.r_c
        )
        return _return

    def _calculateWeights(self, _risk):
        floor = self.getMinRisk()
        cap = self.getMaxRisk()
        if self.sigma_ab > self.sigma_c:
            w = min(
                max((_risk - self.sigma_c) / (self.sigma_ab - self.sigma_c), floor), cap
            )
        else:
            w = cap

        weight_risky = w
        weight_riskless = 1 - w

        weight_a = weight_risky * self.weight_A
        weight_b = weight_risky * self.weight_B
        weight_c = weight_riskless

        return weight_a, weight_b, weight_c


class ParityProcessorDelegate:
    class RiskMode(Enum):
        LOW_RISK = 0
        MEDIUM_RISK = 1
        HIGH_RISK = 2

    def __init__(self, mode):
        if mode == self.RiskMode.LOW_RISK:
            self.risk = 0.15
        elif mode == self.RiskMode.MEDIUM_RISK:
            self.risk = 0.40
        elif mode == self.RiskMode.HIGH_RISK:
            self.risk = 0.60

    def compute_weights(self, parity_line: ParityLine) -> pd.Series:
        weights = parity_line._calculateWeights(self.risk)
        _return = parity_line.convertWeights(weights[0], weights[1], weights[2])
        print(f"Weights: {weights}")
        print(f"Return: {_return}")
        # List all the attributes of the parity line
        print("[PARITY LINE]")
        for attr in dir(parity_line):
            print(f"    {attr}: {getattr(parity_line, attr)}")
        return pd.Series(weights)


class ParityBacktestingProcessor:
    def __init__(
        self,
        portfolio_a: PortfolioPerformance,
        portfolio_b: PortfolioPerformance,
        portfolio_g: PortfolioPerformance,
        parity_lookback_period: int | None = 90,
        mode=ParityProcessorDelegate.RiskMode.LOW_RISK,
    ):
        self.parity_line = ParityLine()

        self.portfolio_a = portfolio_a
        self.portfolio_b = portfolio_b
        self.portfolio_g = portfolio_g

        self.weights = pd.Series(name="Weights")
        self.values = pd.DataFrame(columns=["Portfolio Value"])
        self.holdings = pd.Series(name="Holdings")

        self.parity_lookback_period = parity_lookback_period

        self.delegate = ParityProcessorDelegate(mode)

    def rebalance_line(self, up_to: datetime | None = None):
        assert self.portfolio_a is not None
        assert self.portfolio_b is not None
        assert self.portfolio_g is not None

        self.parity_line.regression(
            self.portfolio_a.up_to(up_to, look_back_period=self.parity_lookback_period),
            self.portfolio_b.up_to(up_to, look_back_period=self.parity_lookback_period),
            self.portfolio_g.up_to(up_to, look_back_period=self.parity_lookback_period),
        )

    def backtest(self, initial_cash: float = 1000000.0):
        # start is the first day of all the portfolios and end is the last day of all the portfolios
        assert self.portfolio_a is not None
        assert self.portfolio_b is not None
        assert self.portfolio_g is not None

        # Remove all NaNs
        self.portfolio_a.portfolio_value = self.portfolio_a.portfolio_value.dropna()
        self.portfolio_b.portfolio_value = self.portfolio_b.portfolio_value.dropna()
        self.portfolio_g.portfolio_value = self.portfolio_g.portfolio_value.dropna()

        start_date = max(
            self.portfolio_a.portfolio_value.index[0],
            self.portfolio_b.portfolio_value.index[0],
            self.portfolio_g.portfolio_value.index[0],
        )
        if self.parity_lookback_period is not None:
            start_date += timedelta(days=self.parity_lookback_period)
        end_date = min(
            self.portfolio_a.portfolio_value.index[-1],
            self.portfolio_b.portfolio_value.index[-1],
            self.portfolio_g.portfolio_value.index[-1],
        )

        self.values.loc[start_date] = initial_cash

        current_date = start_date
        last_rebalance_date = None
        while current_date <= end_date:
            prices = np.array(
                [
                    self.portfolio_a.portfolio_value.loc[current_date],
                    self.portfolio_b.portfolio_value.loc[current_date],
                    self.portfolio_g.portfolio_value.loc[current_date],
                ]
            ).flatten()
            try:
                if last_rebalance_date is None or (
                    self.parity_lookback_period is not None
                    and (current_date - last_rebalance_date).days
                    >= self.parity_lookback_period
                ):
                    if self.parity_lookback_period is not None:
                        self.rebalance_line(current_date)
                    else:
                        self.rebalance_line(end_date)
                    last_rebalance_date = current_date

                    # Convert the ParityLine to weights
                    weights = self.delegate.compute_weights(self.parity_line)
                    assert (
                        sum(weights) - 1 < 1e-5
                    ), f"Weights do not sum to 1: {sum(weights)}"
                    print(f"Weights: {weights}")
                    self.weights.loc[current_date] = weights

                    # Convert the weights to holdings
                    # Last value of the portfolio
                    last_value = (
                        self.values.iloc[-1]["Portfolio Value"]
                        if len(self.values) > 0
                        else initial_cash
                    )
                    if last_value == 0 or np.isnan(last_value):
                        last_value = initial_cash

                    print(
                        f"[Parity] Last value: {last_value}, last iloc: {self.values.iloc[-1]}",
                        f"Current date: {current_date}, last iloc date: {self.values.iloc[-1].name}",
                        "----------",  # Separator, because pandas makes it hard to read
                    )

                    # Allocate the cash to each portfolio

                    self.holdings.loc[current_date] = (
                        last_value * np.array(weights.array) / prices
                    )
                    # Update the portfolio value based on the current prices
                    print(f"Date: {current_date}")
                    print(
                        f"Portfolio A: {self.portfolio_a.portfolio_value.loc[current_date]}"
                    )
                    print(
                        f"Portfolio B: {self.portfolio_b.portfolio_value.loc[current_date]}"
                    )
                    print(
                        f"Portfolio G: {self.portfolio_g.portfolio_value.loc[current_date]}"
                    )
                    print(
                        f"Prices: {prices}, Parity Holdings: {self.holdings.loc[current_date]}"
                    )

            except AssertionError as e:
                import traceback

                traceback.print_exc()
                print(e)

            try:
                _value = prices * self.holdings.iloc[-1]  # Last holdings
                # If the date doesn't exist, this will create a new row
                self.values.at[current_date, "Portfolio Value"] = _value.sum()
                print(f"[Parity] Value updated at {current_date}: {_value.sum()}")
            except Exception as e:
                import traceback

            current_date += timedelta(days=1)

        # Export the holdings to a PortfolioPerformance object
        return PortfolioPerformance(
            portfolio_name="Parity",
            portfolio_value=self.values,
            rebalance_dates=self.holdings.index,
            portfolio_compositions=self.weights,
            portfolio_raw_composition=self.weights,
            portfolio_holdings=self.holdings,
            portfolio_live_weights=self.weights,
        )

    def price_data(self):
        price_data = pd.DataFrame()
        start_date = max(
            self.portfolio_a.portfolio_value.index[0],
            self.portfolio_b.portfolio_value.index[0],
            self.portfolio_g.portfolio_value.index[0],
        )
        if self.parity_lookback_period is not None:
            start_date += timedelta(days=self.parity_lookback_period)
        price_data["Alpha"] = self.portfolio_a.portfolio_value["Portfolio Value"].loc[
            start_date:
        ]
        price_data["Beta"] = self.portfolio_b.portfolio_value["Portfolio Value"].loc[
            start_date:
        ]
        price_data["Gamma"] = self.portfolio_g.portfolio_value["Portfolio Value"].loc[
            start_date:
        ]
        return price_data
