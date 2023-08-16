# Portfolio Allocation

Portfolio Allocation is a Python package for backtesting different portfolio
allocation strategies and visualizing the results in an Excel report.

## Features

- Backtesting multiple portfolio optimization strategies
- Visualization of results in an easy-to-read Excel report

## Basic Usage

Here's a import script to show the exported public functions:

```python
from portfolio_optimization.optimization import (
    HRPOptimization,
    Markowitz,
    BlackLitterman,
    RiskParity,
    FastRiskParity,
    Heuristic,
    RewardToRisk,
    SimpleVolatility,
    VolatilityOfVolatility,
    ValueAtRisk,
    RewardToVaR,
    Combination,
)
from portfolio_optimization.portfolio import Portfolio
from portfolio_optimization.backtesting import Backtest
```

## Testing

You can quickly compare default strategies with a single command:

```shell
./quick_comparison.mjs --rebalance <PERIOD>
```

Where `<PERIOD>` should be replaced with the desired frequency (for example,
`1M`, `1W`, `1D`).
