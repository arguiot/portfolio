import sys
import numpy as np
import json
import rebalancing

inputData = sys.argv[1]
inputDataFormatted = json.loads(inputData)

def saveResults(orderType, assetName, amountAsset, priceAsset, orderSizeAssets, weightAssets, TBDAmount):
    _newAmount = rebalancing.newAmountAsset(amountAsset, priceAsset, weightAssets, TBDAmount)
    _totalbuyOrderAsset = rebalancing.totalbuyOrderAsset(amountAsset, priceAsset, orderSizeAssets, weightAssets, TBDAmount)
    _totalsellOrderAsset = rebalancing.totalsellOrderAsset(amountAsset, priceAsset, orderSizeAssets, weightAssets,TBDAmount)
    _orderSize = rebalancing.orderSize(amountAsset, priceAsset, orderSizeAssets, weightAssets, TBDAmount)

    OrderData = {
        "asset": assetName,
        "assetPrice": priceAsset,
        "balance": amountAsset[0],
        "buyOrders":  _totalbuyOrderAsset.tolist(),
        "sellOrders": _totalsellOrderAsset.tolist(),
        "orderSizeInUSD": _orderSize.tolist(),
        "minNewAmount" : _newAmount[0, 0].tolist(),
        "idealNewAmount" : _newAmount[0,1 ].tolist(),
        "maxNewAmount" : _newAmount[0, 2].tolist(),
        "type": orderType
    }
    
    return OrderData

ordersList = []

for orderData in inputDataFormatted:    
    # assign variables
    nbAsset = 1
    TBDAmount = float(orderData['TBDAmount'])
    weightAssets = np.zeros((nbAsset, 3))
    weightAssets[0,0]= float(orderData['minwit'])
    weightAssets[0,1]= float(orderData['idealwit'])
    weightAssets[0,2]= float(orderData['maxwit'])
    orderSizeAssets = np.zeros((nbAsset, 2))
    orderSizeAssets[0,0] = int(orderData['minsit'])
    orderSizeAssets[0,1] = int(orderData['maxsit'])
    priceAsset = np.zeros(nbAsset)
    priceAsset[0] = float(orderData['p_it'])
    amountAsset= np.zeros(nbAsset)
    amountAsset[0]= float(orderData['currentAmount_it'])

    OrderData = saveResults(orderData['type'], orderData['asset'], amountAsset, priceAsset[0], orderSizeAssets, weightAssets, TBDAmount)
    ordersList.append(OrderData)

print(json.dumps(ordersList))