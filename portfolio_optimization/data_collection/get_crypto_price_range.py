import pandas as pd
import numpy as np
import glob
import os


def get_historical_prices_for_assets(
    assets_list=None,
    folder_path="data/csv",
    interested_columns=["ReferenceRate"],
):
    """
    Returns a pandas DataFrame containing historical prices for a list of assets.

    Parameters:
    assets_list (list): A list of asset names to retrieve prices for. If None, all assets in the specified folder_path will be used.
    folder_path (str): The path to the folder containing the CSV files with the historical prices.
    interested_columns (list): A list of column names to retrieve from the CSV files.

    Returns:
    pandas.DataFrame: A DataFrame containing the historical prices for the specified assets.
    """
    df_dict = {}

    if assets_list is None:
        csv_files = glob.glob(folder_path + "/*.csv")
    else:
        csv_files = [
            folder_path + "/" + asset + ".csv"
            for asset in assets_list
            if os.path.exists(folder_path + "/" + asset + ".csv")
        ]

    for file in csv_files:
        asset_name = os.path.basename(file).split(".")[
            0
        ]  # Extract asset name from file name

        try:
            df = pd.read_csv(file, usecols=["time"] + interested_columns)
            # Rename columns to include asset name for clarity
            rename_dict = {col: f"{asset_name}" for col in interested_columns}
            df.rename(columns=rename_dict, inplace=True)

        except ValueError as e:
            # Missing column in csv, create df with zero
            if "Usecols do not match columns" in str(e):
                df = pd.read_csv(file, usecols=["time"])

                # Add missing columns to dataframe with zero as values
                for col in interested_columns:
                    df[f"{asset_name}"] = 0

        # Rename 'time' column to 'date'
        df["time"] = pd.to_datetime(df["time"])
        df.rename(columns={"time": "date"}, inplace=True)
        df.set_index("date", inplace=True)  # Set date as index
        df_dict[asset_name] = df

    # Join all dataframes on 'time' index
    df_all = pd.concat(df_dict.values(), axis=1, join="inner")

    # Remove rows containing NaN values
    df_all = df_all.dropna()

    # Replace infinite values with NaN
    df_all.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove rows containing infinite values (NaN values after previous replacement)
    df_all = df_all.dropna()

    return df_all
