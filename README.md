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

## Installation

> Make sure you have Python 3.10 or higher installed on your system. Also, make
> sure you have cloned the repository, with submodules!

First, make sure you have CLang installed on your system. If not, you can
install it using the following command:

For macOS:

```shell
brew install llvm
```

For Linux:

```shell
sudo apt-get install clang
```

Then, run the setup script:

```shell
pip install -e .
```

Then, install the required Python packages:

```shell
pip install -r requirements.txt
```

Finally, install Bun:

```shell
curl -fsSL https://bun.sh/install | bash
bun install
```
