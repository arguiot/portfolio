import pandas as pd
import numpy as np
from portfolio_optimization.portfolio.rebalancing import *


def weight_diff(
    holdings: pd.Series, new_target_weights: pd.Series, prices: pd.DataFrame
):
    weight_assets = [
        (
            new_target_weights[i],  # - 0.1 * new_target_weights[i],
            new_target_weights[i],
            new_target_weights[i],  # + 0.1 * new_target_weights[i]
        )
        for i in new_target_weights.index
    ]

    # Order size Assets is a list of tuple such that (minOrderSize, maxOrderSize). For now, min and max are [0, 10000]
    order_size_assets = [(0, 1000) for i in new_target_weights.index]

    orders = totalOrders(
        amountAsset=np.array(holdings),
        priceAsset=np.array(prices),
        orderSizeAssets=np.array(order_size_assets),
        weightAssets=np.array(weight_assets),
        TBDAmount=0,
    )

    sizes = orderSize(
        amountAsset=np.array(holdings),
        priceAsset=np.array(prices),
        orderSizeAssets=np.array(order_size_assets),
        weightAssets=np.array(weight_assets),
        TBDAmount=0,
    )

    trades = orders * sizes
    new_holdings = holdings + trades / prices

    value = new_holdings * prices
    new_weights = value / value.sum()

    return new_weights
