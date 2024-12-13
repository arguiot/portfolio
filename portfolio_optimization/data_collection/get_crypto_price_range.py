import pandas as pd
import numpy as np
import glob
import os
import datetime


def get_historical_prices_for_assets(
    assets_list=None,
    folder_path="data/csv",
    interested_columns=["ReferenceRate"],
    start_date=pd.to_datetime("2021-06-26"),
    end_date=pd.to_datetime("2024-06-25"),
):
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
        asset_name = os.path.basename(file).split(".")[0]

        try:
            # Read the CSV file
            df = pd.read_csv(file)

            # Check if 'time' column exists, if not, try to find a suitable date column
            if "time" not in df.columns:
                date_columns = [
                    col
                    for col in df.columns
                    if "date" in col.lower() or "time" in col.lower()
                ]
                if date_columns:
                    df["time"] = df[date_columns[0]]
                else:
                    print(
                        f"Warning: No suitable date column found for {asset_name}. Skipping this asset."
                    )
                    continue

            # Fill NaN values in ReferenceRate with PriceUSD where available
            if "ReferenceRate" in df.columns and "PriceUSD" in df.columns:
                df["ReferenceRate"] = df["ReferenceRate"].fillna(df["PriceUSD"])

            # Filter out columns not in interested_columns
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

            # Convert 'time' to datetime and set as index
            df["time"] = pd.to_datetime(df["time"])
            df.rename(columns={"time": "date"}, inplace=True)
            df.set_index("date", inplace=True)

            # Reindex df to fill missing dates
            date_range = pd.date_range(start=start_date, end=end_date).drop_duplicates()
            df = df[~df.index.duplicated(keep="first")]  # Remove duplicate labels
            df = df.reindex(date_range, fill_value=np.nan)

            df_dict[asset_name] = df

        except Exception as e:
            print(f"Error processing {asset_name}: {str(e)}")
            continue

    # Join all dataframes on 'date' index
    if df_dict:
        df_all = pd.concat(df_dict.values(), axis=1, join="outer")
        df_all.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df_all
    else:
        print("No valid data found for any asset.")
        return None
