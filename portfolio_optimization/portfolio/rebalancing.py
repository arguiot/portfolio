import numpy as np
import pandas as pd
import cvxpy as cp
from math import ceil


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
    problem.solve(solver=cp.ECOS)

    if problem.status == cp.INFEASIBLE:
        raise Exception(
            f"The problem was not successfully solved! Status: {problem.status}"
        )

    # Get new holdings value
    new_holdings_value = (new_holdings.value * prices).sum()

    assert np.isclose(
        projected_portfolio_val, new_holdings_value, atol=0.00001
    ), f"The new holdings value ({new_holdings_value}) does not match projected portfolio value ({projected_portfolio_val})"

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

    # Calculate current portfolio value and current weights of assets
    portfolio_value = (prices * holdings).sum()
    current_weights = (prices * holdings) / (portfolio_value + external_movement)

    trades = pd.Series(index=shared_index, data=0)

    # Step 0: Sort by deviation from target weights
    deviation = current_weights - new_target_weights
    deviation = deviation.sort_values(ascending=False)

    # Step 1: Find and handle overweight assets
    overweight_assets = current_weights > max_W
    overweight_adjustments = (
        current_weights[overweight_assets] - new_target_weights[overweight_assets]
    ) * (portfolio_value + external_movement)
    cash_from_selling = overweight_adjustments.sum()
    trades[overweight_assets] -= overweight_adjustments / prices

    # Step 2: Add/subtract cash flow from overweight assets
    total_cash = cash_from_selling + external_movement

    # Step 3: Handle negative cash flow
    if total_cash < 0:
        other_assets = current_weights <= max_W
        excess_weights = (
            current_weights[other_assets] - new_target_weights[other_assets]
        )
        excess_weights_assets = excess_weights.sort_values(ascending=False).index
        for asset in excess_weights_assets:
            cash_needed = excess_weights[asset] * portfolio_value
            if (cash_needed + total_cash) < 0:
                # We need to sell this asset
                trades[asset] -= cash_needed / prices[asset]
                total_cash += cash_needed
            else:
                # We only partially sell this asset and break the loop
                trades[asset] += total_cash / prices[asset]
                total_cash = 0  # Update total_cash to 0 after partial sell
                break

    # Step 4: Handle underweight assets
    underweight_assets = current_weights < min_W
    deficit_weights = (
        new_target_weights[underweight_assets] - current_weights[underweight_assets]
    )
    deficit_weights_assets = deficit_weights.sort_values().index[::-1]
    for asset in deficit_weights_assets:
        cash_needed = deficit_weights[asset] * (portfolio_value + external_movement)
        if (cash_needed - total_cash) < 0:
            # We need to buy this asset
            trades[asset] += cash_needed / prices[asset]
            total_cash -= cash_needed
        else:
            # We only partially buy this asset and break the loop
            trades[asset] += total_cash / prices[asset]
            total_cash = 0  # Update total_cash to 0 after partial buy
            break

    # Step 5: Check the cash flow
    start_cash = (prices * holdings).sum() + external_movement
    end_cash = (prices * (holdings + trades)).sum()

    # Update deviation after trades
    current_weights = (prices * (holdings + trades)) / end_cash
    deviation = current_weights - new_target_weights
    cash_to_add = start_cash - end_cash

    deviation = deviation.sort_values(ascending=False)

    if cash_to_add > 0:
        # We need to add cash to the biggest position, if this doesn't violate the max_W constraint. Otherwise, we add cash to the second biggest position, and so on.
        idx = 0
        while cash_to_add > 0 and idx < len(deviation):
            asset = deviation.index[idx]
            potential_trade_value = (
                cash_to_add  # this is the value of tokens we can buy
            )
            potential_trade = (
                potential_trade_value / prices[asset]
            )  # this is the number of tokens we can buy
            if (
                current_weights[asset]
                + (potential_trade * prices[asset] / portfolio_value)
                > max_W[asset]
            ):
                # We can't add the full potential trade, so we add as much as we can
                potential_trade = (
                    (max_W[asset] - current_weights[asset])
                    * portfolio_value
                    / prices[asset]
                )
            trades[asset] += potential_trade
            cash_to_add -= potential_trade * prices[asset]
            idx += 1

    elif cash_to_add < 0:
        # We need to subtract cash from the smallest position, if this doesn't violate the min_W constraint. Otherwise, we subtract cash from the second smallest position, and so on.
        idx = 0
        while cash_to_add < 0 and idx < len(deviation):
            asset = deviation.index[idx]
            potential_trade_value = (
                cash_to_add  # this is the value of tokens we can sell
            )
            potential_trade = potential_trade_value / prices[asset]
            if (
                current_weights[asset]
                - (potential_trade * prices[asset] / portfolio_value)
                < min_W[asset]
            ):
                # We can't subtract the full potential trade, so we subtract as much as we can
                potential_trade = (
                    (current_weights[asset] - min_W[asset])
                    * portfolio_value
                    / prices[asset]
                )
            trades[asset] += potential_trade
            cash_to_add -= potential_trade * prices[asset]
            idx += 1

    total_trade_value = (trades * prices).sum()
    if not np.isclose(total_trade_value, external_movement, atol=1e-8):
        print(
            f"The total trade value is not the same as the external movement. Total trade value: {total_trade_value}, external movement: {external_movement}. Trades: {trades * prices}"
        )
        # Then, we just rebalance to optimal weights, super straightforward
        new_holdings = (
            new_target_weights * (portfolio_value + external_movement) / prices
        )
        trades = new_holdings - holdings

    return trades
