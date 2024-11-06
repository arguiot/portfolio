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
            ]
        )
    )

    h = opt.matrix(np.block([weights_max, -weights_min, cash_in, -cash_out]), tc="d")

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

    wealth_bar_flatten = wealth_bar.flatten()
    chis_flatten = chis.flatten()

    output_wealth_value = np.copy(wealth_bar)

    status = False

    for i in range(n_assets * n_networks):

        try:
            optimal_wealth_value = optimal_wealth_trades_given_boundaries(
                n_assets,
                n_networks,
                weights_max,
                weights_min,
                cash_in,
                cash_out,
                chis_flatten,
                wealth_bar_flatten,
                external_movements,
            )
        except:
            if not status:
                print("[NO SOLUTION]")

            return output_wealth_value, output_weights_min, output_weights_max

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
            if not status:
                print("[NO SOLUTION]")

            return output_wealth_value, output_weights_min, output_weights_max

        sum_abs_deviation_cash = np.abs(
            np.sum(optimal_wealth_value) - np.sum(external_movements)
        )
        error_sum_abs_deviation_cash = sum_abs_deviation_cash > tol_cash

        if error_sum_abs_deviation_cash:
            if not status:
                print("[NO SOLUTION]")

            return output_wealth_value, output_weights_min, output_weights_max

        max_abs_deviation_chis = np.max(np.abs((1 - chis) * optimal_wealth_value))
        error_max_abs_deviation_chis = max_abs_deviation_chis > tol_cash

        if error_max_abs_deviation_chis:
            if not status:
                print("[NO SOLUTION]")

            return output_wealth_value, output_weights_min, output_weights_max

        max_abs_deviation_beta_in = np.max(
            np.sum(optimal_wealth_value, axis=1) - cash_in
        )
        error_max_abs_deviation_beta_in = max_abs_deviation_beta_in > tol_cash

        if error_max_abs_deviation_beta_in:
            if not status:
                print("[NO SOLUTION]")

            return output_wealth_value, output_weights_min, output_weights_max

        max_abs_deviation_beta_out = np.max(
            cash_out - np.sum(optimal_wealth_value, axis=1)
        )
        error_max_abs_deviation_beta_out = max_abs_deviation_beta_out > tol_cash

        if error_max_abs_deviation_beta_out:
            if not status:
                print("[NO SOLUTION]")

            return output_wealth_value, output_weights_min, output_weights_max

        output_wealth_value = np.copy(optimal_wealth_value)

        optimal_wealth_value_flatten = optimal_wealth_value.flatten()
        optimal_wealth_value_flatten[np.array(1 - chis_flatten, dtype=bool)] = np.NaN

        j = np.nanargmin(np.abs(optimal_wealth_value_flatten))
        chis_flatten[j] = 0

        status = True

    return output_wealth_value, output_weights_min, output_weights_max


def optimal_wealth_trades_complete(
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
):

    output_wealth_value, output_weights_min, output_weights_max = optimal_wealth_trades(
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
    )

    output_weights = np.sum(wealth_value + output_wealth_value, axis=0) / np.sum(
        projected_portfolio_values
    )

    relative_deviation = np.divide(
        np.abs(target_weights - output_weights),
        target_weights,
        out=np.float64(np.abs(output_weights) > 1e-10),
        where=(np.abs(target_weights) > 1e-10),
    )

    return output_wealth_value, output_weights, output_weights_min, output_weights_max
