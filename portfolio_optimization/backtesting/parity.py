from datetime import datetime
from datetime import timedelta
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from portfolio_optimization.backtesting import Backtest
from portfolio_optimization.portfolio import Portfolio
from .Backtesting import PortfolioPerformance
from ..optimization.risk_parity import RiskParity
from .delegate import BacktestingDelegate
from enum import Enum


class ParityLine:
    def __init__(self, use_beta=True):
        self.use_beta = use_beta
        self.weight_A = 1.0 if not use_beta else 0.8
        self.weight_B = 0.0 if not use_beta else 0.2
        self.weight_G = 0.0
        self.sigma_a = 0.0
        self.sigma_b = 0.0
        self.sigma_g = 0.0
        self.sigma_ab = 0.0
        self.r_a = 0.0
        self.r_b = 0.0
        self.r_g = 0.0
        self.maxRisk = 0.8
        self.minRisk = 0.0
        self.parity_lookback_period = 90

    def regression(
        self,
        portfolio_a: PortfolioPerformance,
        portfolio_b: PortfolioPerformance | None,
        portfolio_g: PortfolioPerformance,
        override_sigma_g: float | None = None,
    ):
        if self.use_beta and portfolio_b is None:
            raise ValueError("Portfolio B is required when use_beta is True")

        # Utiliser les données tronquées pour RiskParity et les rendements
        portfolio_a_truncated = portfolio_a.up_to(
            look_back_period=self.parity_lookback_period
        )
        portfolio_g_truncated = portfolio_g.up_to(
            look_back_period=self.parity_lookback_period
        )
        if self.use_beta:
            portfolio_b_truncated = portfolio_b.up_to(
                look_back_period=self.parity_lookback_period
            )
        else:
            portfolio_b_truncated = None

        if self.use_beta:
            combined_df = pd.concat(
                [
                    portfolio_a_truncated.portfolio_value["Portfolio Value"].rename(
                        "A"
                    ),
                    portfolio_b_truncated.portfolio_value["Portfolio Value"].rename(
                        "B"
                    ),
                ],
                axis=1,
            )
            rp = RiskParity(df=combined_df)
            _weights = rp.get_weights()
            self.weight_A = _weights.iloc[0]
            self.weight_B = _weights.iloc[1]
        else:
            self.weight_A = 1.0
            self.weight_B = 0.0

        # Calcul des rendements
        self.r_a = self._calculate_return(portfolio_a_truncated)
        if self.use_beta:
            self.r_b = self._calculate_return(portfolio_b_truncated)
        self.r_g = self._calculate_return(portfolio_g_truncated)

        # Calcul des volatilités avec les données complètes jusqu'à 'up_to'
        self.sigma_a = self._calculate_volatility(portfolio_a)
        if self.use_beta:
            self.sigma_b = self._calculate_volatility(portfolio_b)
        self.sigma_g = override_sigma_g or self._calculate_volatility(portfolio_g)

        # Calcul de la volatilité combinée
        if self.use_beta:
            _portfolio_ab_series = (
                portfolio_a.portfolio_value["Portfolio Value"] * self.weight_A
                + portfolio_b.portfolio_value["Portfolio Value"] * self.weight_B
            )
            _portfolio_ab = pd.DataFrame(
                {
                    "Portfolio Value": _portfolio_ab_series,
                }
            )
            portfolio_ab = PortfolioPerformance(
                portfolio_name="Parity",
                portfolio_value=_portfolio_ab,
                rebalance_dates=pd.Series(_portfolio_ab_series.index),
                portfolio_compositions=portfolio_a.portfolio_compositions,
                portfolio_raw_composition=portfolio_a.portfolio_raw_composition,
                portfolio_holdings=portfolio_a.portfolio_holdings,
                portfolio_live_weights=portfolio_a.portfolio_live_weights,
            )
            self.sigma_ab = self._calculate_volatility(portfolio_ab)
        else:
            self.sigma_ab = self.sigma_a

    def _calculate_return(self, portfolio):
        start_date = portfolio.portfolio_value["Portfolio Value"].index[0]
        end_date = portfolio.portfolio_value["Portfolio Value"].index[-1]
        years = (end_date - start_date).days / 365

        # Calcul du rendement (r)
        r = (
            (
                portfolio.portfolio_value["Portfolio Value"].iloc[-1]
                / portfolio.portfolio_value["Portfolio Value"].iloc[0]
            )
            ** (1 / years)
        ) - 1
        return r

    def _calculate_volatility(self, portfolio, up_to=None):
        # Utiliser toutes les données jusqu'à 'up_to' pour calculer la volatilité
        if up_to is not None:
            data = portfolio.portfolio_value["Portfolio Value"].loc[:up_to]
        else:
            data = portfolio.portfolio_value["Portfolio Value"]

        span = (
            self.parity_lookback_period
            if self.parity_lookback_period is not None
            else 90
        )
        portfolio_returns = data.pct_change().dropna()

        rolling_volatility_delta = portfolio_returns.rolling(span).std() * np.sqrt(365)
        rolling_mean = rolling_volatility_delta.rolling(window=7).mean()
        print(f"Rolling mean: {rolling_mean}")
        sigma = rolling_mean.iloc[-1]

        # If sigma is NaN, we compute regular volatility
        if np.isnan(sigma):
            sigma = portfolio_returns.std() * np.sqrt(365)

        return sigma

    def getMinRisk(self):
        return self.minRisk

    def getMaxRisk(self):
        return self.maxRisk

    def convertReturn(self, _return):
        return_riskless = 0.1
        return_risky = 0.5
        sigma_riskless = self.sigma_g
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
            weight_alpha * self.r_a + weight_beta * self.r_b + weight_gamma * self.r_g
        )
        return _return

    def _calculateWeights(self, _risk):
        floor = self.getMinRisk()
        cap = self.getMaxRisk()
        if self.sigma_ab > self.sigma_g:
            w = min(
                max((_risk - self.sigma_g) / (self.sigma_ab - self.sigma_g), floor), cap
            )
        else:
            w = cap

        weight_risky = w
        weight_riskless = 1 - w

        weight_a = weight_risky * self.weight_A
        weight_b = weight_risky * self.weight_B if self.use_beta else 0
        weight_g = weight_riskless

        return (
            (weight_a, weight_g)
            if not self.use_beta
            else (weight_a, weight_b, weight_g)
        )


class ParityProcessorDelegate:
    class RiskMode(Enum):
        LOW_RISK = 0
        MEDIUM_RISK = 1
        HIGH_RISK = 2

    def __init__(self, mode):
        self.mode = mode
        self.threshold = 0.10
        self.override_sigma_g: float | None = None
        if mode == self.RiskMode.LOW_RISK:
            self.risk = 0.15
        elif mode == self.RiskMode.MEDIUM_RISK:
            self.risk = 0.30
        elif mode == self.RiskMode.HIGH_RISK:
            self.risk = 0.45

    def compute_weights(self, parity_line: ParityLine) -> pd.Series:
        # Assign floor and cap risk based on the risk mode
        if self.mode == self.RiskMode.LOW_RISK:  # LOW_RISK
            parity_line.minRisk = 0.10  # 10%
            parity_line.maxRisk = 0.30  # 30%
        elif self.mode == self.RiskMode.MEDIUM_RISK:  # MEDIUM_RISK
            parity_line.minRisk = 0.20  # 25%
            parity_line.maxRisk = 0.80  # 80%
        elif self.mode == self.RiskMode.HIGH_RISK:  # HIGH_RISK
            parity_line.minRisk = 0.40  # 40%
            parity_line.maxRisk = 1.00  # 100%

        weights = parity_line._calculateWeights(self.risk)
        return pd.Series(weights)


class BTCParityProcessorDelegate(ParityProcessorDelegate):
    def __init__(self, mode):
        super().__init__(mode)
        self.threshold = 0.10
        self.override_sigma_g = 0.05
        # Override risk values for BTC
        if mode == self.RiskMode.LOW_RISK:
            self.risk = 0.15
        elif mode == self.RiskMode.MEDIUM_RISK:
            self.risk = 0.35
        elif mode == self.RiskMode.HIGH_RISK:
            self.risk = 0.40

    def compute_weights(self, parity_line: ParityLine) -> pd.Series:
        # Assign floor and cap risk based on the risk mode
        if self.mode == self.RiskMode.LOW_RISK:  # LOW_RISK
            parity_line.minRisk = 0.10  # 10%
            parity_line.maxRisk = 0.30  # 30%
        elif self.mode == self.RiskMode.MEDIUM_RISK:  # MEDIUM_RISK
            parity_line.minRisk = 0.40  # 40%
            parity_line.maxRisk = 1.0  # 100%
        elif self.mode == self.RiskMode.HIGH_RISK:  # HIGH_RISK
            parity_line.minRisk = 0.50  # 50%
            parity_line.maxRisk = 1.00  # 100%

        weights = parity_line._calculateWeights(self.risk)
        return pd.Series(weights)


class ParityBacktestingProcessor:
    def __init__(
        self,
        portfolio_a: PortfolioPerformance,
        portfolio_b: PortfolioPerformance | None,
        portfolio_g: PortfolioPerformance,
        parity_lookback_period: int | None = 90,
        delegate: ParityProcessorDelegate | None = None,
        mode: ParityProcessorDelegate.RiskMode = ParityProcessorDelegate.RiskMode.LOW_RISK,
    ):
        self.use_beta = portfolio_b is not None
        self.parity_line = ParityLine(use_beta=self.use_beta)

        self.portfolio_a = portfolio_a
        self.portfolio_b = portfolio_b
        self.portfolio_g = portfolio_g

        self.weights = pd.Series(name="Weights")
        self.values = pd.DataFrame(columns=["Portfolio Value"])
        self.holdings = pd.Series(name="Holdings")

        self.parity_lookback_period = parity_lookback_period

        self.mode = mode

        self.delegate = (
            delegate if delegate is not None else ParityProcessorDelegate(mode)
        )

        # Initialize the high-risk parity portfolio for volatility calculation
        self.high_risk_parity = pd.DataFrame(columns=["Portfolio Value"])

    def rebalance_line(self, up_to: datetime | None = None):
        assert self.portfolio_a is not None
        assert self.portfolio_g is not None
        if self.use_beta:
            assert self.portfolio_b is not None

        # Obtenir les données tronquées pour RiskParity et les rendements
        portfolio_a_truncated = self.portfolio_a.up_to(up_to)
        portfolio_g_truncated = self.portfolio_g.up_to(up_to)
        if self.use_beta:
            portfolio_b_truncated = self.portfolio_b.up_to(up_to)
        else:
            portfolio_b_truncated = None

        self.parity_line.regression(
            portfolio_a_truncated,
            portfolio_b_truncated if self.use_beta else None,
            portfolio_g_truncated,
            override_sigma_g=self.delegate.override_sigma_g,
        )

    def backtest(self, initial_cash: float = 1000000.0):
        assert self.portfolio_a is not None
        assert self.portfolio_g is not None
        if self.use_beta:
            assert self.portfolio_b is not None

        # Remove all NaNs
        self.portfolio_a.portfolio_value = self.portfolio_a.portfolio_value.dropna()
        if self.use_beta:
            self.portfolio_b.portfolio_value = self.portfolio_b.portfolio_value.dropna()
        self.portfolio_g.portfolio_value = self.portfolio_g.portfolio_value.dropna()

        start_date = max(
            self.portfolio_a.portfolio_value.index[0],
            self.portfolio_g.portfolio_value.index[0],
        )
        if self.use_beta:
            start_date = max(start_date, self.portfolio_b.portfolio_value.index[0])
        if self.parity_lookback_period is not None:
            start_date += timedelta(days=self.parity_lookback_period)
        end_date = min(
            self.portfolio_a.portfolio_value.index[-1],
            self.portfolio_g.portfolio_value.index[-1],
        )
        if self.use_beta:
            end_date = min(end_date, self.portfolio_b.portfolio_value.index[-1])

        self.values.loc[start_date] = initial_cash
        self.high_risk_parity.loc[start_date] = initial_cash

        current_date = start_date
        last_rebalance_date = None
        last_rebalance_vol = 0.0

        vol_history = pd.Series()

        while current_date <= end_date:
            if self.use_beta:
                prices = np.array(
                    [
                        self.portfolio_a.portfolio_value.loc[current_date],
                        self.portfolio_b.portfolio_value.loc[current_date],
                        self.portfolio_g.portfolio_value.loc[current_date],
                    ]
                ).flatten()
            else:
                prices = np.array(
                    [
                        self.portfolio_a.portfolio_value.loc[current_date],
                        self.portfolio_g.portfolio_value.loc[current_date],
                    ]
                ).flatten()

            try:
                # Calculate simple volatility and smooth it over 7 days
                if current_date >= start_date + timedelta(days=7):
                    span = (
                        self.parity_lookback_period
                        if self.parity_lookback_period is not None
                        else 90
                    )
                    portfolio_returns = (
                        self.high_risk_parity["Portfolio Value"].pct_change().dropna()
                    )

                    rolling_volatility_delta = portfolio_returns.rolling(
                        span
                    ).std() * np.sqrt(365)
                    rolling_mean = rolling_volatility_delta.rolling(window=7).mean()
                    print(f"Rolling mean: {rolling_mean}")
                    smoothed_volatility_delta = rolling_mean.iloc[-1]
                else:
                    smoothed_volatility_delta = 0
            except Exception as e:
                print(f"Error calculating volatility: {str(e)}")
                traceback.print_exc()
                print(f"Skipping volatility calculation for {current_date}")
                smoothed_volatility_delta = 0

            vol_history[current_date] = abs(
                smoothed_volatility_delta - last_rebalance_vol
            )
            try:
                if (
                    last_rebalance_date is None
                    or (
                        self.parity_lookback_period is not None
                        and (current_date - last_rebalance_date).days
                        >= self.parity_lookback_period
                    )
                    or abs(smoothed_volatility_delta - last_rebalance_vol)
                    > self.delegate.threshold
                ):
                    if self.parity_lookback_period is not None:
                        self.rebalance_line(current_date)
                    else:
                        self.rebalance_line(end_date)
                    last_rebalance_date = current_date
                    last_rebalance_vol = smoothed_volatility_delta

                    # Convert the ParityLine to weights
                    weights = self.delegate.compute_weights(self.parity_line)
                    assert (
                        sum(weights) - 1 < 1e-5
                    ), f"Weights do not sum to 1: {sum(weights)}"
                    self.weights.loc[current_date] = weights

                    # Calculate weights for high-risk parity portfolio
                    priorMinRisk = self.parity_line.minRisk
                    priorMaxRisk = self.parity_line.maxRisk

                    self.parity_line.minRisk = 0.4
                    self.parity_line.maxRisk = 1.0
                    high_risk_weights = self.parity_line._calculateWeights(0.8)

                    self.parity_line.minRisk = priorMinRisk
                    self.parity_line.maxRisk = priorMaxRisk

                    # Convert the weights to holdings
                    last_value = (
                        self.values.iloc[-1]["Portfolio Value"]
                        if len(self.values) > 0
                        else initial_cash
                    )
                    if last_value == 0 or np.isnan(last_value):
                        last_value = initial_cash

                    # Allocate the cash to each portfolio
                    self.holdings.loc[current_date] = (
                        last_value * np.array(weights.array) / prices
                    )

                    # Calculate holdings for high-risk parity portfolio
                    high_risk_last_value = (
                        self.high_risk_parity.iloc[-1]["Portfolio Value"]
                        if len(self.high_risk_parity) > 0
                        else initial_cash
                    )
                    if high_risk_last_value == 0 or np.isnan(high_risk_last_value):
                        high_risk_last_value = initial_cash

                    high_risk_holdings = (
                        high_risk_last_value * np.array(high_risk_weights) / prices
                    )

            except AssertionError as e:
                import traceback

                traceback.print_exc()
                print(e)

            try:
                _value = prices * self.holdings.iloc[-1]  # Last holdings
                self.values.at[current_date, "Portfolio Value"] = _value.sum()

                # Calculate value for high-risk parity portfolio
                high_risk_value = prices * high_risk_holdings
                self.high_risk_parity.at[current_date, "Portfolio Value"] = (
                    high_risk_value.sum()
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(e)

            current_date += timedelta(days=1)

        # Save a plot of the volatility history in the output folder
        plt.clf()  # Clear the current figure
        vol_history.plot(title="Volatility History")
        plt.legend()  # Add a legend to the plot
        prefix = "btc_" if isinstance(self.delegate, BTCParityProcessorDelegate) else ""
        plt.savefig(f"out/{prefix}{self.mode}_volatility_history.png")
        plt.close()  # Close the figure to free up memory

        # Export the holdings to a PortfolioPerformance object
        return PortfolioPerformance(
            portfolio_name="Parity",
            portfolio_value=self.values,
            rebalance_dates=pd.Series(self.holdings.index),
            portfolio_compositions=self.weights,
            portfolio_raw_composition=self.weights,
            portfolio_holdings=self.holdings,
            portfolio_live_weights=self.weights,
        )

    def price_data(self):
        price_data = pd.DataFrame()
        start_date = max(
            self.portfolio_a.portfolio_value.index[0],
            self.portfolio_g.portfolio_value.index[0],
        )
        if self.use_beta:
            start_date = max(start_date, self.portfolio_b.portfolio_value.index[0])
        if self.parity_lookback_period is not None:
            start_date += timedelta(days=self.parity_lookback_period)
        price_data["Alpha"] = self.portfolio_a.portfolio_value["Portfolio Value"].loc[
            start_date:
        ]
        if self.use_beta:
            price_data["Beta"] = self.portfolio_b.portfolio_value[
                "Portfolio Value"
            ].loc[start_date:]
        price_data["Gamma"] = self.portfolio_g.portfolio_value["Portfolio Value"].loc[
            start_date:
        ]
        return price_data
