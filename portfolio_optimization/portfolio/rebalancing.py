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


def deterministic_optimal_rebalancing(
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

    # Calculate current portfolio value and weights
    current_portfolio_value = (holdings * prices).sum()
    current_weights = (holdings * prices) / current_portfolio_value

    # Find overweight assets and adjust them to target, calculating cash surplus
    cash_surplus = 0
    for asset in shared_index:
        if isinstance(max_W, float):
            max_W_asset = max_W
        else:
            max_W_asset = max_W[asset]
        if current_weights[asset] > max_W_asset:
            cost_to_adjust = (
                current_weights[asset] - new_target_weights[asset]
            ) * current_portfolio_value
            cash_surplus += cost_to_adjust
            holdings[asset] -= cost_to_adjust / prices[asset]

    # Update cash surplus based on external movement
    cash_surplus += external_movement

    # Calculate current weights
    current_portfolio_value = (holdings * prices).sum()
    current_weights = (holdings * prices) / current_portfolio_value

    # Sort underweight assets by deviation from minimum weight
    underweight_assets = {
        asset: min_W[asset] - current_weights[asset]
        if not isinstance(min_W, float)
        else min_W - current_weights[asset]
        for asset in shared_index
        if current_weights[asset]
        < (min_W[asset] if not isinstance(min_W, float) else min_W)
    }
    underweight_assets = dict(
        sorted(underweight_assets.items(), key=lambda item: item[1], reverse=True)
    )

    # Allocate cash surplus to underweight assets until no more cash or all assets at minimum weight
    for asset, weight_diff in underweight_assets.items():
        cash_needed = weight_diff * current_portfolio_value
        cash_to_allocate = min(cash_needed, cash_surplus)
        holdings[asset] += cash_to_allocate / prices[asset]
        cash_surplus -= cash_to_allocate

        if cash_surplus <= 0:
            break

    # Calculate difference in asset holdings
    diff = holdings - holdings.values
    return diff
