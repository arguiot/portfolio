import unittest
from portfolio_optimization.data_collection import *
from pandas.testing import assert_frame_equal
import tempfile
import shutil
import os


class TestHistoricalPrices(unittest.TestCase):
    def setUp(self):
        # Set up temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Delete temporary directory and all files within it
        shutil.rmtree(self.temp_dir)

    def test_empty_asset_list(self):
        # With no assets specified, all csv files in the directory should be used
        df = pd.DataFrame(
            {"time": ["2020-01-01", "2020-01-02"], "ReferenceRate": [0, 1]}
        )
        df.to_csv(os.path.join(self.temp_dir, "asset1.csv"), index=False)
        expected_df = df.rename(
            columns={"time": "date", "ReferenceRate": "asset1"}
        ).set_index("date")

        output_df = get_historical_prices_for_assets(folder_path=self.temp_dir)
        assert_frame_equal(output_df, expected_df)

    def test_single_asset(self):
        # Test with a single specified asset
        df = pd.DataFrame(
            {"time": ["2020-01-01", "2020-01-02"], "ReferenceRate": [0, 1]}
        )
        df.to_csv(os.path.join(self.temp_dir, "asset1.csv"), index=False)
        expected_df = df.rename(
            columns={"time": "date", "ReferenceRate": "asset1"}
        ).set_index("date")

        output_df = get_historical_prices_for_assets(
            ["asset1"], folder_path=self.temp_dir
        )
        assert_frame_equal(output_df, expected_df)

    def test_missing_column(self):
        # Test with a csv file that doesn't contain all the interested columns
        df = pd.DataFrame(
            {"time": ["2020-01-01", "2020-01-02"], "SomeOtherRate": [0, 1]}
        )
        df.to_csv(os.path.join(self.temp_dir, "asset1.csv"), index=False)
        expected_df = (
            df.drop(columns=["SomeOtherRate"])
            .assign(asset1=0)
            .rename(columns={"time": "date"})
            .set_index("date")
        )

        output_df = get_historical_prices_for_assets(
            ["asset1"], folder_path=self.temp_dir
        )
        assert_frame_equal(output_df, expected_df)
