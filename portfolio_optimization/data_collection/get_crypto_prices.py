import requests
import json

def get_crypto_prices(crypto_ids):
    url = 'https://api.coingecko.com/api/v3/simple/price'
    params = {'ids': ','.join(crypto_ids), 'vs_currencies': 'usd'}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return json.loads(response.text)
    else:
        return None
