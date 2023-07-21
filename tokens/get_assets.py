import requests
import os


def read_tickers_from_file(file_path):
    with open(file_path, "r") as file:
        return [
            line.lower() for line in file.read().splitlines()
        ]  # convert to lower case


def get_coingecko_assets():
    response = requests.get("https://api.coingecko.com/api/v3/coins/list")
    response.raise_for_status()
    return {coin["symbol"]: coin["id"] for coin in response.json()}


def get_tickers():
    # assets = get_coingecko_assets()  # get CoinGecko assets

    # Print the amount of tickers in the assets dict
    # print("Total tickers:", len(assets))

    # Get absolute paths of ticker files
    script_dir = os.path.dirname(os.path.realpath(__file__))
    high_risk_path = os.path.join(script_dir, "high_risk.txt")
    med_risk_path = os.path.join(script_dir, "medium_risk.txt")
    low_risk_path = os.path.join(script_dir, "low_risk.txt")

    # Print the amount of tickers in each file
    print("High risk tickers:", len(read_tickers_from_file(high_risk_path)))
    print("Medium risk tickers:", len(read_tickers_from_file(med_risk_path)))
    print("Low risk tickers:", len(read_tickers_from_file(low_risk_path)))

    # high_risk_tickers = [
    #     assets.get(symbol)
    #     for symbol in read_tickers_from_file(high_risk_path)
    #     if assets.get(symbol)
    # ]
    # medium_risk_tickers = [
    #     assets.get(symbol)
    #     for symbol in read_tickers_from_file(med_risk_path)
    #     if assets.get(symbol)
    # ]
    # low_risk_tickers = [
    #     assets.get(symbol)
    #     for symbol in read_tickers_from_file(low_risk_path)
    #     if assets.get(symbol)
    # ]

    return {
        # "high_risk": high_risk_tickers,
        # "medium_risk": medium_risk_tickers,
        # "low_risk": low_risk_tickers,
        "high_risk_tickers": read_tickers_from_file(high_risk_path),
        "medium_risk_tickers": read_tickers_from_file(med_risk_path),
        "low_risk_tickers": read_tickers_from_file(low_risk_path),
    }


if __name__ == "__main__":
    ticker_assets = get_tickers()
    print(ticker_assets)
