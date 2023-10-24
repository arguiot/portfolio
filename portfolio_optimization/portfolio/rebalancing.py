import numpy as np
import pandas as pd
import cvxpy as cp


def optimize_trades(
    holdings: pd.Series,
    new_target_weights: pd.Series,
    prices: pd.Series,
    min_W: pd.Series | float,
    max_W: pd.Series | float,
    external_movement: float,
):
    shared_index = prices.index.intersection(new_target_weights.index)

    prices = prices.reindex(shared_index)
    holdings = holdings.reindex(shared_index)
    new_target_weights = new_target_weights.reindex(shared_index)

    if not isinstance(min_W, float):
        min_W = min_W.reindex(shared_index)
    if not isinstance(max_W, float):
        max_W = max_W.reindex(shared_index)

    current_value = (holdings * prices).sum()
    n_assets = len(holdings)

    projected_portfolio_val = current_value + external_movement

    # Define new holdings as a variable
    new_holdings = cp.Variable(n_assets)

    min_W = min_W if isinstance(min_W, float) else min_W.values
    max_W = max_W if isinstance(max_W, float) else max_W.values

    # Constraint: new total assets value should not exceed the adjusted current value
    constraints = [
        cp.sum(cp.multiply(new_holdings, prices.values)) == projected_portfolio_val,
        cp.multiply(min_W, projected_portfolio_val)
        <= cp.multiply(new_holdings, prices.values),
        cp.multiply(new_holdings, prices.values)
        <= cp.multiply(max_W, projected_portfolio_val),
    ]

    # Relative weights of the new portfolio
    new_weights = cp.multiply(new_holdings, prices.values) / projected_portfolio_val

    objective = cp.Minimize(
        cp.sum_squares(new_weights - new_target_weights.values)
        # + cp.sum_squares(new_holdings - holdings.values)
    )

    problem = cp.Problem(objective, constraints)
    problem.solve(warm_start=True)

    print(new_weights.value - new_target_weights.values)
    print(problem.status)

    if problem.status == cp.INFEASIBLE:
        raise Exception(
            f"The problem was not successfully solved! Status: {problem.status}"
        )

    # Get new holdings value
    new_holdings_value = (new_holdings.value * prices).sum()

    assert np.isclose(
        projected_portfolio_val, new_holdings_value, atol=0.001
    ), "The new holdings value does not match projected portfolio value"

    diff = new_holdings.value - holdings.values
    diff = pd.Series(diff, index=prices.index)
    return diff
