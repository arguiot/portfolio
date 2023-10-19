import numpy as np


def currentAmountAsset(_amountAsset, _priceAsset):
    return _amountAsset * _priceAsset


def currentTotalAmountAsset(_amountAsset, _priceAsset):
    return np.sum(_amountAsset * _priceAsset)


def depositINDAsset(_TBDAmount):
    return 1 if _TBDAmount >= 0 else 0


def withdrawINDAsset(_TBDAmount):
    return 1 - depositINDAsset(_TBDAmount)


def newAmountAsset(_amountAsset, _priceAsset, _weightAssets, _TBDAmount):
    nbAsset = np.shape(_amountAsset)[0]

    _newAmount = np.zeros((nbAsset, 3))

    _currentTotalAmount = currentTotalAmountAsset(_amountAsset, _priceAsset)

    _newAmount[:, 0] = np.minimum(
        (_currentTotalAmount + _TBDAmount) * _weightAssets[:, 0],
        (_currentTotalAmount + _TBDAmount) * _weightAssets[:, 2])

    _newAmount[:, 1] = (_currentTotalAmount + _TBDAmount) * _weightAssets[:, 1]
    _newAmount[:, 2] = np.maximum(
        (_currentTotalAmount + _TBDAmount) * _weightAssets[:, 0],
        (_currentTotalAmount + _TBDAmount) * _weightAssets[:, 2])
    return _newAmount


def minMaxCurrentDifAsset(_amountAsset, _priceAsset, _weightAssets,
                          _TBDAmount):
    nbAsset = np.shape(_amountAsset)[0]
    _minMaxCurrentDif = np.zeros(nbAsset)

    _depositIND = depositINDAsset(_TBDAmount)

    _withdrawIND = withdrawINDAsset(_TBDAmount)

    _newAmountMinAsset = newAmountAsset(_amountAsset, _priceAsset,
                                        _weightAssets, _TBDAmount)[:, 0]

    _newAmountMaxAsset = newAmountAsset(_amountAsset, _priceAsset,
                                        _weightAssets, _TBDAmount)[:, 2]

    _currentAmount = currentAmountAsset(_amountAsset, _priceAsset)

    _minMaxCurrentDif = _depositIND * (_newAmountMaxAsset -
                                       _currentAmount) + _withdrawIND * (
                                           _newAmountMinAsset - _currentAmount)

    return _minMaxCurrentDif


def rebalanceDeltaAsset(_amountAsset, _priceAsset, _weightAssets, _TBDAmount):
    nbAsset = np.shape(_amountAsset)[0]
    _rebalanceDelta = np.zeros(nbAsset)

    _depositIND = depositINDAsset(_TBDAmount)

    _withdrawIND = withdrawINDAsset(_TBDAmount)

    _newAmountMinAsset = newAmountAsset(_amountAsset, _priceAsset,
                                        _weightAssets, _TBDAmount)[:, 0]

    _newAmountMaxAsset = newAmountAsset(_amountAsset, _priceAsset,
                                        _weightAssets, _TBDAmount)[:, 1]

    _currentAmount = currentAmountAsset(_amountAsset, _priceAsset)

    _rebalanceDelta = _depositIND * np.minimum(
        _newAmountMaxAsset - _currentAmount,
        np.zeros(nbAsset)) + _withdrawIND * np.maximum(
            _newAmountMinAsset - _currentAmount, np.zeros(nbAsset))
    return _rebalanceDelta


def rebalanceDeltaTotalAsset(_amountAsset, _priceAsset, _weightAssets,
                             _TBDAmount):

    return np.sum(
        rebalanceDeltaAsset(_amountAsset, _priceAsset, _weightAssets,
                            _TBDAmount))


def rebalanceMinSizeDeltaAsset(_amountAsset, _priceAsset, _orderSizeAssets, _weightAssets, _TBDAmount):
    nbAsset = np.shape(_amountAsset)[0]
    _rebalanceDelta = rebalanceDeltaAsset(_amountAsset, _priceAsset,
                                          _weightAssets, _TBDAmount)

    _rebalanceMinSizeDelta = np.zeros(nbAsset)

    for i in range(nbAsset):
        _rebalanceMinSizeDelta[i] = abs(_rebalanceDelta[i]) if (
            _rebalanceDelta[i] < _orderSizeAssets[i, 0]) else 0

    return _rebalanceMinSizeDelta


def rebalanceMinSizeDeltaTotalAsset(_amountAsset, _priceAsset,
                                    _orderSizeAssets, _weightAssets, _TBDAmount):

    return np.sum(
        rebalanceMinSizeDeltaAsset(_amountAsset, _priceAsset, _orderSizeAssets, _weightAssets, _TBDAmount))


def buyINDAsset(_amountAsset, _priceAsset, orderSizeAssets, _weightAssets,
                _TBDAmount): 
    nbAsset = np.shape(_amountAsset)[0]
    _buyINDAsset = np.zeros(nbAsset)

    _minMaxCurrentDif = minMaxCurrentDifAsset(_amountAsset, _priceAsset,
                                              _weightAssets, _TBDAmount)

    for i in range(nbAsset):

        _buyINDAsset[i] = 1 if _minMaxCurrentDif[i] >= 0 else 0

    return _buyINDAsset


def totalbuyOrderAsset(_amountAsset, _priceAsset, _orderSizeAssets,
                       _weightAssets, _TBDAmount):
    _buyIND = buyINDAsset(_amountAsset, _priceAsset, _orderSizeAssets,
                          _weightAssets, _TBDAmount)

    return np.sum(_buyIND)


def totalsellOrderAsset(_amountAsset, _priceAsset, orderSizeAssets, weightAssets,
                     _TBDAmount):
    nbAsset = np.shape(_amountAsset)[0]
    return nbAsset - totalbuyOrderAsset(
        _amountAsset, _priceAsset, orderSizeAssets, weightAssets, _TBDAmount)


def maxToMinRankAsset(_amountAsset, _priceAsset, _weightAssets, _TBDAmount):
    nbAsset = np.shape(_amountAsset)[0]
    _minMaxCurrentDif = np.copy(
        minMaxCurrentDifAsset(_amountAsset, _priceAsset, _weightAssets,
                              _TBDAmount))

    _minMaxCurrentDifSorted = np.sort(_minMaxCurrentDif)[::-1]

    _maxToMinRank = np.zeros(nbAsset)

    for i in range(nbAsset):

        _list = np.where(_minMaxCurrentDifSorted == _minMaxCurrentDif[i])[0]

        _rank = _list[0]

        _count = np.size(_list)

        _maxToMinRank[i] = _rank + _count - 1

    return _maxToMinRank


def minToMaxRankAsset(_amountAsset, _priceAsset, _weightAssets, _TBDAmount):
    nbAsset = np.shape(_amountAsset)[0]
    _minToMaxRank = np.zeros(nbAsset)

    _minMaxCurrentDif = np.copy(
        minMaxCurrentDifAsset(_amountAsset, _priceAsset, _weightAssets,
                              _TBDAmount))

    _minMaxCurrentDifSorted = np.sort(_minMaxCurrentDif)

    for i in range(nbAsset):

        _list = np.where(_minMaxCurrentDifSorted == _minMaxCurrentDif[i])[0]

        _rank = _list[0]

        _count = np.size(_list)

        _minToMaxRank[i] = _rank + _count - 1
    return _minToMaxRank


def assetCapRankAsset(_amountAsset, _priceAsset, _orderSizeAssets,
                      _weightAssets, _TBDAmount):
    nbAsset = np.shape(_amountAsset)[0]
    _assetCapRank = np.zeros(nbAsset)

    _maxToMinRank = maxToMinRankAsset(_amountAsset, _priceAsset, _weightAssets,
                                      _TBDAmount)

    _minToMaxRank = minToMaxRankAsset(_amountAsset, _priceAsset, _weightAssets,
                                      _TBDAmount)

    _buyIND = buyINDAsset(_amountAsset, _priceAsset, _orderSizeAssets,
                          _weightAssets, _TBDAmount)

    _totalsellOrderst = totalsellOrderAsset(_amountAsset, _priceAsset,
                                         _orderSizeAssets, _weightAssets,
                                         _TBDAmount)

    _assetCapRank = _minToMaxRank + _buyIND * (_maxToMinRank - _minToMaxRank +
                                               _totalsellOrderst)

    return np.int32(_assetCapRank)


def rawCapFilledAsset(_amountAsset, _priceAsset, _orderSizeAssets,
                      _weightAssets, _TBDAmount):
    nbAsset = np.shape(_amountAsset)[0]
    rawCapFilled = np.zeros(nbAsset)

    _minMaxCurrentDif = minMaxCurrentDifAsset(_amountAsset, _priceAsset,
                                              _weightAssets, _TBDAmount)

    _assetCapRank = assetCapRankAsset(_amountAsset, _priceAsset,
                                      _orderSizeAssets, _weightAssets,
                                      _TBDAmount)

    for i in range(nbAsset):

        rawCapFilled[i] = np.sum(_minMaxCurrentDif[0:_assetCapRank[i]])

    return rawCapFilled


def capToFillAsset(_amountAsset, _priceAsset, _orderSizeAssets, _weightAssets,
                   _TBDAmount):
    _depositIND = depositINDAsset(_TBDAmount)

    _minMaxCurrentDif = minMaxCurrentDifAsset(_amountAsset, _priceAsset,
                                              _weightAssets, _TBDAmount)

    _rebalanceMinSizeDeltaTotal = rebalanceMinSizeDeltaTotalAsset(
        _amountAsset, _priceAsset, _orderSizeAssets, _weightAssets, _TBDAmount)

    _rawCapFilled = rawCapFilledAsset(_amountAsset, _priceAsset,
                                      _orderSizeAssets, _weightAssets,
                                      _TBDAmount)

    _depositTerm = _depositIND * np.minimum(
        _minMaxCurrentDif,
        (_TBDAmount + _rebalanceMinSizeDeltaTotal - np.minimum(
            (_TBDAmount + _rebalanceMinSizeDeltaTotal), _rawCapFilled)))

    _withdrawIND = withdrawINDAsset(_TBDAmount)

    _rebalanceDeltaTotal = rebalanceDeltaTotalAsset(_amountAsset, _priceAsset,
                                                    _weightAssets, _TBDAmount)

    _withdrawTerm = _withdrawIND * np.minimum(
        _minMaxCurrentDif, (_TBDAmount + _rebalanceDeltaTotal - np.minimum(
            (_TBDAmount + _rebalanceDeltaTotal), _rawCapFilled)))
    return _depositTerm + _withdrawTerm


def minOrderAsset(_amountAsset, _priceAsset, _orderSizeAssets, _weightAssets,
                  _TBDAmount):
    nbAsset = np.shape(_amountAsset)[0]
    _minOrder = np.zeros(nbAsset)

    _capToFillAsset = capToFillAsset(_amountAsset, _priceAsset,
                                     _orderSizeAssets, _weightAssets,
                                     _TBDAmount)

    for i in range(nbAsset):

        _minOrder[i] = 1 if abs(_capToFillAsset[i]) >= _orderSizeAssets[i, 0] else 0
    return _minOrder


def additionalOrdersAsset(_amountAsset, _priceAsset, _orderSizeAssets,
                          _weightAssets, _TBDAmount):
    _capToFillAsset = np.absolute(
        capToFillAsset(_amountAsset, _priceAsset, _orderSizeAssets,
                       _weightAssets, _TBDAmount))
    return np.floor(_capToFillAsset / _orderSizeAssets[:, 1])


def totalOrders(amountAsset, _priceAsset, _orderSizeAssets, _weightAssets,
                _TBDAmount):

    return minOrderAsset(amountAsset, _priceAsset, _orderSizeAssets,
                         _weightAssets, _TBDAmount) + additionalOrdersAsset(
                             amountAsset, _priceAsset, _orderSizeAssets,
                             _weightAssets, _TBDAmount)


def orderSize(_amountAsset, _priceAsset, _orderSizeAssets, _weightAssets,
              _TBDAmount):
    _capToFillAsset = capToFillAsset(_amountAsset, _priceAsset,
                                     _orderSizeAssets, _weightAssets,
                                     _TBDAmount)
    _additionalOrders = additionalOrdersAsset(_amountAsset, _priceAsset,
                                              _orderSizeAssets, _weightAssets,
                                              _TBDAmount)
    return _capToFillAsset / (_additionalOrders + 1)
