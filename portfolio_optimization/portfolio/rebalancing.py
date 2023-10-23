import numpy as np
import pandas as pd
import cvxpy as cp


def optimize_trades(
    holdings: pd.Series,
    new_target_weights: pd.Series,
    prices: pd.Series,
    min_W: pd.Series | float,
    max_W: pd.Series | float,
    external_movement: float,  # external movement of the portfolio
    l1_reg: float = 1.0,
):
    current_value = (holdings * prices).sum()
    n_assets = len(holdings)  # number of assets

    # Value of current holdings, adjust by external_movement
    V = current_value + external_movement

    # Variable for the optimization problem: the number of units to buy/sell of each asset
    trades = cp.Variable(n_assets)

    # New holdings will be old holdings plus trades
    new_holdings = holdings.values + trades

    min_W = min_W if isinstance(min_W, float) else min_W.values
    max_W = max_W if isinstance(max_W, float) else max_W.values
    # Constraints of the optimization problem
    constraints = [
        # The sum of the portfolio values should be equal to the new portfolio value (i.e., we add or withdraw money)
        V == cp.sum(cp.multiply(new_holdings, prices.values)),
        # Each new holding should be within its corresponding min and max weight
        cp.multiply(new_holdings, prices.values)
        >= cp.multiply(min_W, V),  # note: the nonnegative elementwise multiplication
        cp.multiply(new_holdings, prices.values)
        <= cp.multiply(
            max_W, V
        ),  # assumes min_W and max_W are arrays of the same dimensionality
    ]

    # Define the objective of the optimization problem: minimize the squared difference
    # between the new weights and the target weights
    objective = cp.Minimize(
        cp.sum_squares(
            cp.multiply(new_holdings, prices.values) / V - new_target_weights.values
        )
        + l1_reg * cp.norm(trades, 1)
    )

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Check if the problem was successfully solved
    if problem.status != cp.OPTIMAL:
        raise Exception("The problem was not successfully solved!")

    # New Holdings value
    new_holdings_value = (new_holdings.value * prices).sum()

    assert np.isclose(
        V, new_holdings_value
    ), "The new holdings value is not equal to the new portfolio value"

    # Return the optimal trades
    diff = new_holdings.value - holdings.values
    diff = pd.Series(diff, index=prices.index)
    return diff / prices
