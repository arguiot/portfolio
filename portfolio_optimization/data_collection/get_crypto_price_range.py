from datetime import datetime, timezone, timedelta
import requests
import pandas as pd
import json


def get_crypto_price_range(id, start_date, end_date):
    # Convert dates to UNIX timestamps
    start_date_timestamp = datetime.timestamp(start_date)
    end_date_timestamp = datetime.timestamp(end_date)

    url = f"https://api.coingecko.com/api/v3/coins/{id}/market_chart/range"
    parameters = {
        "vs_currency": "usd",
        "from": start_date_timestamp,
        "to": end_date_timestamp,
    }

    response = requests.get(url, params=parameters)

    if response.status_code == 200:
        return json.loads(response.text)
    else:
        return None


def get_historical_prices_for_assets(asset_list, start_date, end_date):
    # Initialize an empty dict of dataframes
    df_dict = dict()

    for asset in asset_list:
        # Get historical price of asset
        data = get_crypto_price_range(asset, start_date, end_date)
        if data is not None:
            prices = data["prices"]
            # Convert data to DataFrame
            df = pd.DataFrame(prices, columns=["time", asset])
            # Convert timestamp to datetime
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            # Set time as index
            df.set_index("time", inplace=True)
            # Resampling the time series data to 1H intervals
            df = df.resample("1H").mean()
            df_dict[asset] = df
        else:
            print(f"Failed to fetch data for {asset}")

    # Concatenate all dataframes
    df_all = pd.concat(df_dict.values(), axis=1)

    return df_all
