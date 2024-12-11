import numpy as np
import cvxopt as opt
from cvxopt import solvers


def optimal_wealth_trades_given_boundaries(
    n_assets,
    n_networks,
    weights_max,
    weights_min,
    cash_in,
    cash_out,
    chis_flatten,
    wealth_flatten,
    wealth_bar_flatten,
    external_movements,
):
    P = opt.matrix(np.eye(n_assets * n_networks))
    q = opt.matrix(-wealth_bar_flatten.T)

    G = opt.matrix(
        np.vstack(
            [
                np.block([np.eye(n_assets) for j in range(n_networks)]),
                np.block([-np.eye(n_assets) for j in range(n_networks)]),
                np.repeat(np.eye(n_networks), n_assets, axis=1),
                np.repeat(-np.eye(n_networks), n_assets, axis=1),
                np.block([-np.eye(n_assets * n_networks)]),
            ]
        )
    )

    h = opt.matrix(
        np.block([weights_max, -weights_min, cash_in, -cash_out, wealth_flatten]),
        tc="d",
    )

    chis_filtered = np.diag((1 - chis_flatten))
    non_zero_rows = np.any(chis_filtered != 0, axis=1)
    chis_filtered = chis_filtered[non_zero_rows]

    A = opt.matrix(np.vstack([np.ones(n_assets * n_networks), chis_filtered]))

    b = opt.matrix(
        np.block(
            [np.sum(external_movements), np.zeros(np.int64(np.sum(1 - chis_flatten)))]
        ),
        tc="d",
    )

    output = solvers.qp(P, q, G, h, A, b)["x"]
    output = np.reshape(
        output,
        [
            -1,
        ],
    )

    return np.reshape(output, [n_networks, n_assets])


def optimal_wealth_trades_given_network(
    n_assets,
    n_networks,
    alphas,
    target_weights,
    wealth_value,
    chis,
    current_chis,
    beta_out,
    beta_in,
    projected_portfolio_values,
    external_movements,
    current_network,
):

    tol_weight = 1e-10
    tol_cash = 1e-2

    epsilon = 1e-4
    bar_alphas = alphas * (1 - epsilon)

    wealth_bar = (
        chis * target_weights * np.reshape(projected_portfolio_values, [-1, 1])
        - wealth_value
    )
    wealth_bar += (
        chis
        * np.sum(
            (1 - chis)
            * target_weights
            * np.reshape(projected_portfolio_values, [-1, 1]),
            axis=0,
        )
        / np.sum(chis, axis=0)
    )

    total_projected_portfolio_values = np.sum(projected_portfolio_values)

    weights_max = target_weights * total_projected_portfolio_values * (
        1 + bar_alphas
    ) - np.sum(wealth_value, axis=0)
    weights_min = target_weights * total_projected_portfolio_values * (
        1 - bar_alphas
    ) - np.sum(wealth_value, axis=0)

    output_weights_max = target_weights * (1 + alphas)
    output_weights_min = target_weights * (1 - alphas)

    cash_in = external_movements + beta_in
    cash_out = external_movements - beta_out

    wealth_flatten = wealth_value.flatten()
    wealth_bar_flatten = wealth_bar.flatten()
    current_chis_flatten = current_chis.flatten()

    output_wealth_value = np.copy(wealth_bar)

    status = False

    mask_network = np.zeros((n_networks, n_assets), dtype=bool)
    mask_network[current_network] = True
    mask_network_flatten = mask_network.flatten()

    current_chis_flatten_try = current_chis_flatten.copy()
    n_available_assets = np.int64(chis.sum(axis=1)[current_network])

    for i in range(n_available_assets + 1):

        try:
            optimal_wealth_value = optimal_wealth_trades_given_boundaries(
                n_assets,
                n_networks,
                weights_max,
                weights_min,
                cash_in,
                cash_out,
                current_chis_flatten_try,
                wealth_flatten,
                wealth_bar_flatten,
                external_movements,
            )
        except:
            return status, output_wealth_value, current_chis_flatten

        optimal_weights = (
            np.sum(wealth_value + optimal_wealth_value, axis=0)
            / total_projected_portfolio_values
        )

        abs_relative_deviation = np.divide(
            np.abs(target_weights - optimal_weights),
            target_weights,
            out=np.float64(np.abs(optimal_weights) > tol_weight),
            where=(np.abs(target_weights) > tol_weight),
        )

        max_abs_relative_deviation = np.max(abs_relative_deviation - alphas)
        error_max_abs_relative_deviation = max_abs_relative_deviation > tol_weight

        if error_max_abs_relative_deviation:
            return status, output_wealth_value, current_chis_flatten

        sum_abs_deviation_cash = np.abs(
            np.sum(optimal_wealth_value) - np.sum(external_movements)
        )
        error_sum_abs_deviation_cash = sum_abs_deviation_cash > tol_cash

        if error_sum_abs_deviation_cash:
            return status, output_wealth_value, current_chis_flatten

        max_abs_deviation_chis = np.max(np.abs((1 - chis) * optimal_wealth_value))
        error_max_abs_deviation_chis = max_abs_deviation_chis > tol_cash

        if error_max_abs_deviation_chis:
            return status, output_wealth_value, current_chis_flatten

        max_abs_deviation_beta_in = np.max(
            np.sum(optimal_wealth_value, axis=1) - cash_in
        )
        error_max_abs_deviation_beta_in = max_abs_deviation_beta_in > tol_cash

        if error_max_abs_deviation_beta_in:
            return status, output_wealth_value, current_chis_flatten

        max_abs_deviation_beta_out = np.max(
            cash_out - np.sum(optimal_wealth_value, axis=1)
        )
        error_max_abs_deviation_beta_out = max_abs_deviation_beta_out > tol_cash

        if error_max_abs_deviation_beta_out:
            return status, output_wealth_value, current_chis_flatten

        output_wealth_value = np.copy(optimal_wealth_value)
        current_chis_flatten = np.copy(current_chis_flatten_try)

        if i < n_available_assets:
            optimal_wealth_value_flatten = optimal_wealth_value.flatten()
            optimal_wealth_value_flatten[
                np.array(1 - current_chis_flatten, dtype=bool)
            ] = np.NaN
            optimal_wealth_value_flatten[
                np.array(1 - mask_network_flatten, dtype=bool)
            ] = np.NaN

            index = np.nanargmin(np.abs(optimal_wealth_value_flatten))
            current_chis_flatten_try[index] = 0

        status = True

    return status, output_wealth_value, current_chis_flatten


def optimal_wealth_trades(
    n_assets,
    n_networks,
    alphas,
    target_weights,
    wealth_value,
    chis,
    beta_out,
    beta_in,
    projected_portfolio_values,
    external_movements,
    priority_queue,
):

    tol_weight = 1e-10
    tol_cash = 1e-2

    epsilon = 1e-4
    bar_alphas = alphas * (1 - epsilon)

    wealth_bar = (
        chis * target_weights * np.reshape(projected_portfolio_values, [-1, 1])
        - wealth_value
    )
    wealth_bar += (
        chis
        * np.sum(
            (1 - chis)
            * target_weights
            * np.reshape(projected_portfolio_values, [-1, 1]),
            axis=0,
        )
        / np.sum(chis, axis=0)
    )

    output_weights_max = target_weights * (1 + alphas)
    output_weights_min = target_weights * (1 - alphas)

    output_wealth_value = np.copy(wealth_bar)
    current_chis_flatten = chis.flatten()

    for j in range(n_networks):

        current_network = np.flip(priority_queue)[j]

        status, output_wealth_value, current_chis_flatten = (
            optimal_wealth_trades_given_network(
                n_assets,
                n_networks,
                alphas,
                target_weights,
                wealth_value,
                chis,
                np.reshape(current_chis_flatten, [n_networks, n_assets]),
                beta_out,
                beta_in,
                projected_portfolio_values,
                external_movements,
                current_network,
            )
        )

    return status, output_wealth_value, output_weights_min, output_weights_max


def optimal_wealth_trades_complete_given_bridge_tolerances(
    n_assets,
    n_networks,
    alphas,
    target_weights,
    wealth_value,
    chis,
    beta_out,
    beta_in,
    projected_portfolio_values,
    external_movements,
    priority_queue,
):

    status, output_wealth_value, output_weights_min, output_weights_max = (
        optimal_wealth_trades(
            n_assets,
            n_networks,
            alphas,
            target_weights,
            wealth_value,
            chis,
            beta_out,
            beta_in,
            projected_portfolio_values,
            external_movements,
            priority_queue,
        )
    )

    output_weights = np.sum(wealth_value + output_wealth_value, axis=0) / np.sum(
        projected_portfolio_values
    )

    output_bridges = np.sum(output_wealth_value, axis=1) - external_movements

    relative_deviation = np.divide(
        np.abs(target_weights - output_weights),
        target_weights,
        out=np.float64(np.abs(output_weights) > 1e-10),
        where=(np.abs(target_weights) > 1e-10),
    )

    return (
        status,
        output_wealth_value,
        output_bridges,
        output_weights,
        output_weights_min,
        output_weights_max,
    )


def optimal_wealth_trades_complete(
    n_assets,
    n_networks,
    alphas,
    target_weights,
    wealth_value,
    chis,
    projected_portfolio_values,
    external_movements,
    priority_queue,
    beta_in=None,
    beta_out=None,
):
    if beta_in is None:
        beta_in = np.zeros(n_networks)
    if beta_out is None:
        beta_out = np.zeros(n_networks)
    (
        status,
        output_wealth_value,
        output_bridges,
        output_weights,
        output_weights_min,
        output_weights_max,
    ) = optimal_wealth_trades_complete_given_bridge_tolerances(
        n_assets,
        n_networks,
        alphas,
        target_weights,
        wealth_value,
        chis,
        beta_out,
        beta_in,
        projected_portfolio_values,
        external_movements,
        priority_queue,
    )

    if status:
        return (
            status,
            output_wealth_value,
            output_bridges,
            output_weights,
            output_weights_min,
            output_weights_max,
        )

    else:

        inflows = np.zeros(n_networks)
        inflows[external_movements > 0] = external_movements[external_movements > 0]

        outflows = np.zeros(n_networks)
        outflows[external_movements < 0] = external_movements[external_movements < 0]

        beta_in_temp = np.sum(inflows) - inflows
        beta_out_temp = np.abs(np.sum(outflows) - outflows)

        beta_in = np.maximum(beta_in_temp, np.max(beta_out_temp))
        beta_out = np.maximum(beta_out_temp, np.max(beta_in_temp))

        (
            status,
            output_wealth_value,
            output_bridges,
            output_weights,
            output_weights_min,
            output_weights_max,
        ) = optimal_wealth_trades_complete_given_bridge_tolerances(
            n_assets,
            n_networks,
            alphas,
            target_weights,
            wealth_value,
            chis,
            beta_out,
            beta_in,
            projected_portfolio_values,
            external_movements,
            priority_queue,
        )

        return (
            status,
            output_wealth_value,
            output_bridges,
            output_weights,
            output_weights_min,
            output_weights_max,
        )
