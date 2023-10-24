import numpy as np
import pandas as pd
from numpy.typing import NDArray


def currentAmountAsset(amountAsset: NDArray, priceAsset: NDArray) -> NDArray:
    """
    Function to calculate the current amount of asset.

    Parameters:
    amountAsset : array
        The amount of asset.
    priceAsset : array
        The price of asset.

    Returns:
    array
        The current amount of asset.
    """
    return amountAsset * priceAsset


def currentTotalAmountAsset(
    amountAsset: NDArray[np.float64], priceAsset: NDArray[np.float64]
) -> float:
    """
    Function to calculate the current total amount of asset.

    Parameters:
    amountAsset : array
        The amount of asset.
    priceAsset : array
        The price of asset.

    Returns:
    float
        The current total amount of asset.
    """
    return float(np.sum(amountAsset * priceAsset))


def depositINDAsset(TBDAmount: float) -> int:
    """
    Function to calculate the deposit indicator of asset.

    Parameters:
    TBDAmount : float
        The amount to be determined.

    Returns:
    int
        The deposit indicator of asset.
    """
    return 1 if TBDAmount >= 0 else 0


def withdrawINDAsset(TBDAmount: float) -> int:
    """
    Function to calculate the withdrawal indicator of asset.

    Parameters:
    TBDAmount : float
        The amount to be determined.

    Returns:
    int
        The withdrawal indicator of asset.
    """
    return 1 - depositINDAsset(TBDAmount)


def newAmountAsset(
    amountAsset: NDArray,
    priceAsset: NDArray,
    weightAssets: NDArray,
    TBDAmount: float,
) -> NDArray:
    """
    Function to calculate the new amount of asset.

    Parameters:
    amountAsset : array
        The amount of asset.
    priceAsset : array
        The price of asset.
    weightAssets : array
        The weight of assets.
    TBDAmount : float
        The amount to be determined.

    Returns:
    array
        The new amount of asset.
    """
    nbAsset = np.shape(amountAsset)[0]

    newAmount = np.zeros((nbAsset, 3))

    currentTotalAmount = currentTotalAmountAsset(amountAsset, priceAsset)

    newAmount[:, 0] = np.minimum(
        (currentTotalAmount + TBDAmount) * weightAssets[:, 0],
        (currentTotalAmount + TBDAmount) * weightAssets[:, 2],
    )

    newAmount[:, 1] = (currentTotalAmount + TBDAmount) * weightAssets[:, 1]
    newAmount[:, 2] = np.maximum(
        (currentTotalAmount + TBDAmount) * weightAssets[:, 0],
        (currentTotalAmount + TBDAmount) * weightAssets[:, 2],
    )
    return newAmount


def minMaxCurrentDifAsset(
    amountAsset: NDArray,
    priceAsset: NDArray,
    weightAssets: NDArray,
    TBDAmount: float,
) -> NDArray:
    """
    Function to calculate the minimum and maximum current difference of asset.

    Parameters:
    amountAsset : array
        The amount of asset.
    priceAsset : array
        The price of asset.
    weightAssets : array
        The weight of assets.
    TBDAmount : float
        The amount to be determined.

    Returns:
    array
        The minimum and maximum current difference of asset.
    """
    nbAsset = np.shape(amountAsset)[0]
    minMaxCurrentDif = np.zeros(nbAsset)

    depositIND = depositINDAsset(TBDAmount)

    withdrawIND = withdrawINDAsset(TBDAmount)

    newAmountMinAsset = newAmountAsset(
        amountAsset, priceAsset, weightAssets, TBDAmount
    )[:, 0]

    newAmountMaxAsset = newAmountAsset(
        amountAsset, priceAsset, weightAssets, TBDAmount
    )[:, 2]

    currentAmount = currentAmountAsset(amountAsset, priceAsset)

    minMaxCurrentDif = depositIND * (
        newAmountMaxAsset - currentAmount
    ) + withdrawIND * (newAmountMinAsset - currentAmount)

    return minMaxCurrentDif


def rebalanceDeltaAsset(
    amountAsset: NDArray,
    priceAsset: NDArray,
    weightAssets: NDArray,
    TBDAmount: float,
) -> NDArray:
    """
    Function to calculate the rebalance delta of asset.

    Parameters:
    amountAsset : array
        The amount of asset.
    priceAsset : array
        The price of asset.
    weightAssets : array
        The weight of assets.
    TBDAmount : float
        The amount to be determined.

    Returns:
    array
        The rebalance delta of asset.
    """
    nbAsset = np.shape(amountAsset)[0]
    rebalanceDelta = np.zeros(nbAsset)

    depositIND = depositINDAsset(TBDAmount)

    withdrawIND = withdrawINDAsset(TBDAmount)

    newAmountMinAsset = newAmountAsset(
        amountAsset, priceAsset, weightAssets, TBDAmount
    )[:, 0]

    newAmountMaxAsset = newAmountAsset(
        amountAsset, priceAsset, weightAssets, TBDAmount
    )[:, 1]

    currentAmount = currentAmountAsset(amountAsset, priceAsset)

    rebalanceDelta = depositIND * np.minimum(
        newAmountMaxAsset - currentAmount, np.zeros(nbAsset)
    ) + withdrawIND * np.maximum(newAmountMinAsset - currentAmount, np.zeros(nbAsset))
    return rebalanceDelta


def rebalanceDeltaTotalAsset(
    amountAsset: NDArray,
    priceAsset: NDArray,
    weightAssets: NDArray,
    TBDAmount: float,
) -> float:
    """
    Function to calculate the total rebalance delta of asset.

    Parameters:
    amountAsset : array
        The amount of asset.
    priceAsset : array
        The price of asset.
    weightAssets : array
        The weight of assets.
    TBDAmount : float
        The amount to be determined.

    Returns:
    float
        The total rebalance delta of asset.
    """
    return np.sum(rebalanceDeltaAsset(amountAsset, priceAsset, weightAssets, TBDAmount))


def rebalanceMinSizeDeltaAsset(
    amountAsset: NDArray,
    priceAsset: NDArray,
    orderSizeAssets: NDArray,
    weightAssets: NDArray,
    TBDAmount: float,
) -> NDArray:
    """
    Function to calculate the minimum size delta of asset for rebalancing.

    Parameters:
    amountAsset : array
        The amount of asset.
    priceAsset : array
        The price of asset.
    orderSizeAssets : array
        The order size of assets.
    weightAssets : array
        The weight of assets.
    TBDAmount : float
        The amount to be determined.

    Returns:
    array
        The minimum size delta of asset for rebalancing.
    """
    nbAsset = np.shape(amountAsset)[0]
    rebalanceDelta = rebalanceDeltaAsset(
        amountAsset, priceAsset, weightAssets, TBDAmount
    )

    rebalanceMinSizeDelta = np.zeros(nbAsset)

    for i in range(nbAsset):
        rebalanceMinSizeDelta[i] = (
            abs(rebalanceDelta[i]) if (rebalanceDelta[i] < orderSizeAssets[i, 0]) else 0
        )

    return rebalanceMinSizeDelta


def rebalanceMinSizeDeltaTotalAsset(
    amountAsset: NDArray,
    priceAsset: NDArray,
    orderSizeAssets: NDArray,
    weightAssets: NDArray,
    TBDAmount: float,
) -> float:
    """
    Function to calculate the total minimum size delta of asset for rebalancing.

    Parameters:
    amountAsset : array
        The amount of asset.
    priceAsset : array
        The price of asset.
    orderSizeAssets : array
        The order size of assets.
    weightAssets : array
        The weight of assets.
    TBDAmount : float
        The amount to be determined.

    Returns:
    float
        The total minimum size delta of asset for rebalancing.
    """
    return np.sum(
        rebalanceMinSizeDeltaAsset(
            amountAsset, priceAsset, orderSizeAssets, weightAssets, TBDAmount
        )
    )


def buyINDAsset(
    amountAsset: NDArray,
    priceAsset: NDArray,
    orderSizeAssets: NDArray,
    weightAssets: NDArray,
    TBDAmount: float,
) -> NDArray:
    """
    Function to determine the buy indicator of asset.

    Parameters:
    amountAsset : array
        The amount of asset.
    priceAsset : array
        The price of asset.
    orderSizeAssets : array
        The order size of assets.
    weightAssets : array
        The weight of assets.
    TBDAmount : float
        The amount to be determined.

    Returns:
    array
        The buy indicator of asset.
    """
    nbAsset = np.shape(amountAsset)[0]
    buyINDAsset = np.zeros(nbAsset)

    minMaxCurrentDif = minMaxCurrentDifAsset(
        amountAsset, priceAsset, weightAssets, TBDAmount
    )

    for i in range(nbAsset):
        buyINDAsset[i] = 1 if minMaxCurrentDif[i] >= 0 else 0

    return buyINDAsset


def totalbuyOrderAsset(
    amountAsset: NDArray,
    priceAsset: NDArray,
    orderSizeAssets: NDArray,
    weightAssets: NDArray,
    TBDAmount: float,
) -> float:
    """
    Function to calculate the total buy order of asset.

    Parameters:
    amountAsset : array
        The amount of asset.
    priceAsset : array
        The price of asset.
    orderSizeAssets : array
        The order size of assets.
    weightAssets : array
        The weight of assets.
    TBDAmount : float
        The amount to be determined.

    Returns:
    float
        The total buy order of asset.
    """
    buyIND = buyINDAsset(
        amountAsset, priceAsset, orderSizeAssets, weightAssets, TBDAmount
    )

    return np.sum(buyIND)


def totalsellOrderAsset(
    amountAsset: NDArray,
    priceAsset: NDArray,
    orderSizeAssets: NDArray,
    weightAssets: NDArray,
    TBDAmount: float,
) -> float:
    """
    Function to calculate the total sell order of asset.

    Parameters:
    amountAsset : array
        The amount of asset.
    priceAsset : array
        The price of asset.
    orderSizeAssets : array
        The order size of assets.
    weightAssets : array
        The weight of assets.
    TBDAmount : float
        The amount to be determined.

    Returns:
    float
        The total sell order of asset.
    """
    nbAsset = np.shape(amountAsset)[0]
    return nbAsset - totalbuyOrderAsset(
        amountAsset, priceAsset, orderSizeAssets, weightAssets, TBDAmount
    )


def maxToMinRankAsset(_amountAsset, _priceAsset, _weightAssets, _TBDAmount):
    nbAsset = np.shape(_amountAsset)[0]
    _minMaxCurrentDif = np.copy(
        minMaxCurrentDifAsset(_amountAsset, _priceAsset, _weightAssets, _TBDAmount)
    )

    _minMaxCurrentDifSorted = _minMaxCurrentDif.argsort()[::-1][:nbAsset]
    return _minMaxCurrentDifSorted


def minToMaxRankAsset(_amountAsset, _priceAsset, _weightAssets, _TBDAmount):
    # nbAsset = np.shape(_amountAsset)[0]
    # _minToMaxRank = np.zeros(nbAsset)

    _minMaxCurrentDif = np.copy(
        minMaxCurrentDifAsset(_amountAsset, _priceAsset, _weightAssets, _TBDAmount)
    )

    _minMaxCurrentDifSorted = _minMaxCurrentDif.argsort()
    return _minMaxCurrentDifSorted


def assetCapRankAsset(
    amountAsset: NDArray,
    priceAsset: NDArray,
    orderSizeAssets: NDArray,
    weightAssets: NDArray,
    TBDAmount: float,
) -> np.int32:
    """
    Function to calculate the asset cap rank.

    Parameters:
    amountAsset : array
        The amount of asset.
    priceAsset : array
        The price of asset.
    orderSizeAssets : array
        The order size of assets.
    weightAssets : array
        The weight of assets.
    TBDAmount : float
        The amount to be determined.

    Returns:
    array
        The asset cap rank.
    """
    nbAsset = np.shape(amountAsset)[0]
    assetCapRank = np.zeros(nbAsset)

    maxToMinRank = maxToMinRankAsset(amountAsset, priceAsset, weightAssets, TBDAmount)

    minToMaxRank = minToMaxRankAsset(amountAsset, priceAsset, weightAssets, TBDAmount)

    buyIND = buyINDAsset(
        amountAsset, priceAsset, orderSizeAssets, weightAssets, TBDAmount
    )

    totalsellOrderst = totalsellOrderAsset(
        amountAsset, priceAsset, orderSizeAssets, weightAssets, TBDAmount
    )

    assetCapRank = minToMaxRank + buyIND * (
        maxToMinRank - minToMaxRank + totalsellOrderst
    )

    return np.int32(assetCapRank)


def rawCapFilledAsset(
    amountAsset: NDArray,
    priceAsset: NDArray,
    orderSizeAssets: NDArray,
    weightAssets: NDArray,
    TBDAmount: float,
) -> NDArray:
    """
    Function to calculate the raw cap filled of asset.

    Parameters:
    amountAsset : array
        The amount of asset.
    priceAsset : array
        The price of asset.
    orderSizeAssets : array
        The order size of assets.
    weightAssets : array
        The weight of assets.
    TBDAmount : float
        The amount to be determined.

    Returns:
    array
        The raw cap filled of asset.
    """
    nbAsset = np.shape(amountAsset)[0]
    rawCapFilled = np.zeros(nbAsset)

    minMaxCurrentDif = minMaxCurrentDifAsset(
        amountAsset, priceAsset, weightAssets, TBDAmount
    )

    assetCapRank = assetCapRankAsset(
        amountAsset, priceAsset, orderSizeAssets, weightAssets, TBDAmount
    )

    for i in range(nbAsset):
        rawCapFilled[i] = np.sum(minMaxCurrentDif[0 : assetCapRank[i]])

    return rawCapFilled


def capToFillAsset(
    amountAsset: NDArray,
    priceAsset: NDArray,
    orderSizeAssets: NDArray,
    weightAssets: NDArray,
    TBDAmount: float,
) -> NDArray:
    """
    Function to calculate the cap to fill of asset.

    Parameters:
    amountAsset : array
        The amount of asset.
    priceAsset : array
        The price of asset.
    orderSizeAssets : array
        The order size of assets.
    weightAssets : array
        The weight of assets.
    TBDAmount : float
        The amount to be determined.

    Returns:
    array
        The cap to fill of asset.
    """
    depositIND = depositINDAsset(TBDAmount)

    minMaxCurrentDif = minMaxCurrentDifAsset(
        amountAsset, priceAsset, weightAssets, TBDAmount
    )

    rebalanceMinSizeDeltaTotal = rebalanceMinSizeDeltaTotalAsset(
        amountAsset, priceAsset, orderSizeAssets, weightAssets, TBDAmount
    )

    rawCapFilled = rawCapFilledAsset(
        amountAsset, priceAsset, orderSizeAssets, weightAssets, TBDAmount
    )

    depositTerm = depositIND * np.minimum(
        minMaxCurrentDif,
        (
            TBDAmount
            + rebalanceMinSizeDeltaTotal
            - np.minimum((TBDAmount + rebalanceMinSizeDeltaTotal), rawCapFilled)
        ),
    )

    withdrawIND = withdrawINDAsset(TBDAmount)

    rebalanceDeltaTotal = rebalanceDeltaTotalAsset(
        amountAsset, priceAsset, weightAssets, TBDAmount
    )

    withdrawTerm = withdrawIND * np.minimum(
        minMaxCurrentDif,
        (
            TBDAmount
            + rebalanceDeltaTotal
            - np.minimum((TBDAmount + rebalanceDeltaTotal), rawCapFilled)
        ),
    )
    return depositTerm + withdrawTerm


def minOrderAsset(
    amountAsset: NDArray,
    priceAsset: NDArray,
    orderSizeAssets: NDArray,
    weightAssets: NDArray,
    TBDAmount: float,
) -> NDArray:
    """
    Function to calculate the minimum order of asset.

    Parameters:
    amountAsset : array
        The amount of asset.
    priceAsset : array
        The price of asset.
    orderSizeAssets : array
        The order size of assets.
    weightAssets : array
        The weight of assets.
    TBDAmount : float
        The amount to be determined.

    Returns:
    array
        The minimum order of asset.
    """
    nbAsset = np.shape(amountAsset)[0]
    minOrder = np.zeros(nbAsset)

    _capToFillAsset = capToFillAsset(
        amountAsset, priceAsset, orderSizeAssets, weightAssets, TBDAmount
    )

    for i in range(nbAsset):
        minOrder[i] = 1 if abs(_capToFillAsset[i]) >= orderSizeAssets[i, 0] else 0
    return minOrder


def additionalOrdersAsset(
    amountAsset: NDArray,
    priceAsset: NDArray,
    orderSizeAssets: NDArray,
    weightAssets: NDArray,
    TBDAmount: float,
) -> NDArray:
    """
    Function to calculate the additional orders of asset.

    Parameters:
    amountAsset : array
        The amount of asset.
    priceAsset : array
        The price of asset.
    orderSizeAssets : array
        The order size of assets.
    weightAssets : array
        The weight of assets.
    TBDAmount : float
        The amount to be determined.

    Returns:
    array
        The additional orders of asset.
    """
    _capToFillAsset = np.absolute(
        capToFillAsset(
            amountAsset, priceAsset, orderSizeAssets, weightAssets, TBDAmount
        )
    )
    return np.floor(_capToFillAsset / orderSizeAssets[:, 1])


def totalOrders(
    amountAsset: NDArray,
    priceAsset: NDArray,
    orderSizeAssets: NDArray,
    weightAssets: NDArray,
    TBDAmount: float,
) -> NDArray:
    """
    Function to calculate the total orders of asset.

    Parameters:
    amountAsset : array
        The amount of asset.
    priceAsset : array
        The price of asset.
    orderSizeAssets : array
        The order size of assets.
    weightAssets : array
        The weight of assets.
    TBDAmount : float
        The amount to be determined.

    Returns:
    array
        The total orders of asset.
    """
    return minOrderAsset(
        amountAsset, priceAsset, orderSizeAssets, weightAssets, TBDAmount
    ) + additionalOrdersAsset(
        amountAsset, priceAsset, orderSizeAssets, weightAssets, TBDAmount
    )


def orderSize(
    amountAsset: NDArray,
    priceAsset: NDArray,
    orderSizeAssets: NDArray,
    weightAssets: NDArray,
    TBDAmount: float,
) -> NDArray:
    """
    Function to calculate the order size of asset.

    Parameters:
    amountAsset : array
        The amount of asset.
    priceAsset : array
        The price of asset.
    orderSizeAssets : array
        The order size of assets.
    weightAssets : array
        The weight of assets.
    TBDAmount : float
        The amount to be determined.

    Returns:
    array
        The order size of asset.
    """
    _capToFillAsset = capToFillAsset(
        amountAsset, priceAsset, orderSizeAssets, weightAssets, TBDAmount
    )
    additionalOrders = additionalOrdersAsset(
        amountAsset, priceAsset, orderSizeAssets, weightAssets, TBDAmount
    )
    return _capToFillAsset / (additionalOrders + 1)
