import pandas as pd
import numpy as np
import glob
import os
import datetime


def get_historical_prices_for_assets(
    assets_list=None,
    folder_path="data/csv",
    interested_columns=["ReferenceRate"],
    time_range=datetime.timedelta(days=90),
):
    """
    ...

    Parameters:
    ...
    time_range (datetime.timedelta): Time range to retrieve prices for. By default, the last 90 days are used.

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
            df = pd.read_csv(file)
            # Filter out columns not in interested_columns. If column is not in df, make the value NaN. Keep 'time' column.
            df = df[
                ["time"]
                + [
                    col
                    for col in interested_columns
                    if col in df.columns or col == "ReferenceRate"
                ]
            ]
            # Create missing columns with NaN values
            for col in interested_columns:
                if col not in df.columns and col != "ReferenceRate":
                    df[col] = np.nan

            # Rename columns to include asset name for clarity
            rename_dict = {
                col: f"{asset_name}{'' if col == 'ReferenceRate' else f'_{col}'}"
                for col in interested_columns
            }
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

        # If time_range is not set, create one for 3 years
        max_date = df.index.max()
        if not time_range and max_date - time_range > df.index.min():
            time_range = max_date - df.index.min()

        date_range = pd.date_range(start=max_date - time_range, end=max_date)

        # Reindex df to fill missing dates
        df = df.reindex(date_range, fill_value=np.nan)

        df_dict[asset_name] = df

    # Join all dataframes on 'time' index
    df_all = pd.concat(df_dict.values(), axis=1, join="inner")

    # Replace infinite values with NaN
    df_all.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df_all
