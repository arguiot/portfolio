import os
import csv
import requests
from datetime import datetime

# Get the CoinGecko API key from environment variable
api_key = os.getenv("COINGECKO_API")

if not api_key:
    api_key = "CG-cM98zKfpLQ94ssiAp2SGh34g"
    # raise ValueError("COINGECKO_API environment variable not set")


def fetch_token_data(token_id):
    base_url = "https://pro-api.coingecko.com/api/v3"
    endpoint = f"/coins/{token_id}/market_chart?vs_currency=usd&days=max&interval=daily"

    headers = {"X-Cg-Pro-Api-Key": api_key}

    response = requests.get(base_url + endpoint, headers=headers)

    response.raise_for_status()

    data = response.json()
    return data


def process_data(raw_data):
    processed_data = []

    prices = raw_data["prices"]
    market_caps = raw_data["market_caps"]

    for price, market_cap in zip(prices, market_caps):
        timestamp = price[0]
        date = datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d")
        price_usd = price[1]
        market_cap_usd = market_cap[1]

        processed_data.append(
            {"time": date, "ReferenceRate": price_usd, "CapMrktEstUSD": market_cap_usd}
        )

    return processed_data


def write_to_csv(data, filename):
    with open(filename, "w", newline="") as csvfile:
        fieldnames = ["time", "ReferenceRate", "CapMrktEstUSD"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            writer.writerow(row)


def main():
    token_id = input("Enter the CoinGecko token ID (e.g., bitcoin): ")
    ticker = input("Enter the ticker (e.g., BTC): ")
    output_file = f"{ticker}.csv"

    print(f"Fetching data for {token_id}...")
    raw_data = fetch_token_data(token_id)

    print("Processing data...")
    processed_data = process_data(raw_data)

    print(f"Writing data to {output_file}...")
    write_to_csv(processed_data, output_file)

    print("Done!")


if __name__ == "__main__":
    main()
