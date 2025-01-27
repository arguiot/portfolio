import os
import csv
import requests
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Get the CoinGecko API key from environment variable
api_key = os.getenv("COINGECKO_API")

if not api_key:
    api_key = "CG-cM98zKfpLQ94ssiAp2SGh34g"
    # raise ValueError("COINGECKO_API environment variable not set")


def fetch_markets_page(page):
    """Fetch a single page of the coins/markets endpoint."""
    base_url = "https://pro-api.coingecko.com/api/v3"
    endpoint = "/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 250,
        "page": page,
        "sparkline": "false",
    }
    headers = {"X-Cg-Pro-Api-Key": api_key}
    response = requests.get(base_url + endpoint, headers=headers, params=params)
    response.raise_for_status()
    return response.json()


def build_symbol_mapping(max_pages=10):
    """Build a symbol-to-id mapping using the coins/markets endpoint.
    If multiple coins have the same symbol, choose the one with the highest market cap.
    """
    symbol_map = {}

    for page in range(1, max_pages + 1):
        print(f"Fetching markets page {page}...")
        data = fetch_markets_page(page)
        if not data:
            break

        for coin in data:
            symbol = coin["symbol"].lower()
            coin_id = coin["id"]
            market_cap = coin.get("market_cap", 0)

            # If this symbol hasn't been seen or found a coin with bigger market cap
            if (
                symbol not in symbol_map
                or market_cap > symbol_map[symbol]["market_cap"]
            ):
                symbol_map[symbol] = {"id": coin_id, "market_cap": market_cap}

        # If less than 250 returned, likely end of pages
        if len(data) < 250:
            break

    # Convert symbol_map to a simple symbol->id dict
    return {s: info["id"] for s, info in symbol_map.items()}


def fetch_range_data(token_id, from_ts, to_ts):
    """Fetch market data for a given token_id from 'from_ts' to 'to_ts' (UNIX timestamps, in seconds)."""
    base_url = "https://pro-api.coingecko.com/api/v3"
    endpoint = f"/coins/{token_id}/market_chart/range"
    params = {"vs_currency": "usd", "from": from_ts, "to": to_ts}
    headers = {"X-Cg-Pro-Api-Key": api_key}
    response = requests.get(base_url + endpoint, headers=headers, params=params)
    response.raise_for_status()
    return response.json()


def fetch_three_year_data(token_id):
    """Fetch the last 3 years of daily data in chunks of 90 days using the market_chart/range endpoint."""

    # Calculate the timestamps for the last 3 years
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=5 * 365)  # approx 5 years
    end_ts = int(end_date.timestamp())
    start_ts = int(start_date.timestamp())

    chunk_days = 90
    chunk_seconds = chunk_days * 24 * 60 * 60

    data_prices = []
    data_market_caps = []

    current_start = start_ts
    while current_start < end_ts:
        current_end = current_start + chunk_seconds
        if current_end > end_ts:
            current_end = end_ts

        raw_data = fetch_range_data(token_id, current_start, current_end)

        if "prices" in raw_data:
            data_prices.extend(raw_data["prices"])
        if "market_caps" in raw_data:
            data_market_caps.extend(raw_data["market_caps"])

        current_start = current_end

    # Create daily aggregated dictionaries
    daily_prices = {}
    daily_caps = {}

    # Process prices
    for ts, price in data_prices:
        date_str = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
        if date_str not in daily_prices:
            daily_prices[date_str] = price
        # If we already have a price for this date, we'll keep the first one

    # Process market caps
    for ts, cap in data_market_caps:
        date_str = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
        if date_str not in daily_caps:
            daily_caps[date_str] = cap
        # If we already have a market cap for this date, we'll keep the first one

    # Combine the data
    combined_data = []
    for date_str in sorted(daily_prices.keys()):
        combined_data.append(
            {
                "time": date_str,
                "ReferenceRate": daily_prices[date_str],
                "CapMrktEstUSD": daily_caps.get(
                    date_str
                ),  # Use .get() in case market cap is missing
            }
        )

    return combined_data


def write_to_csv(data, filename):
    output_dir = os.path.join("..", "data", "csv")
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["time", "ReferenceRate", "CapMrktEstUSD"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            writer.writerow(row)


def main():
    # Build symbol->id mapping using the markets endpoint
    print("Building symbol mapping from markets data...")
    symbol_to_id = build_symbol_mapping(max_pages=10)

    # Read all tickers from the three risk files
    risk_files = ["high_risk.txt", "medium_risk.txt", "low_risk.txt", "arthur.txt"]
    tickers = set()  # use a set to avoid duplicates
    for filename in risk_files:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    ticker = line.strip()
                    if ticker:
                        tickers.add(ticker)
        else:
            print(f"Warning: {filename} does not exist.")

    # Prepare tasks for parallel execution
    tasks = []
    results = {}

    # We'll use ThreadPoolExecutor to fetch data in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        for ticker in tickers:
            symbol_lower = ticker.lower()
            if symbol_lower not in symbol_to_id:
                print(
                    f"Warning: No CoinGecko ID found for ticker '{ticker}'. Skipping."
                )
                continue

            token_id = symbol_to_id[symbol_lower]
            print(f"Queueing data fetch for ticker '{ticker}' (id: {token_id})...")
            future = executor.submit(fetch_three_year_data, token_id)
            results[future] = (ticker, token_id)

        # Retrieve results as they complete
        for future in as_completed(results):
            ticker, token_id = results[future]
            try:
                data = future.result()
                output_file = f"{ticker}.csv"
                print(f"Writing data for {ticker} to ../data/csv/{output_file}...")
                write_to_csv(data, output_file)
            except Exception as e:
                print(f"Error fetching data for {ticker} (id: {token_id}): {e}")

    print("Done!")


if __name__ == "__main__":
    main()
