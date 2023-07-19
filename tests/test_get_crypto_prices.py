import unittest
from portfolio_optimization.data_collection import get_crypto_prices

class TestGetData(unittest.TestCase):
    def test_get_crypto_prices(self):
        crypto_ids = ['bitcoin', 'ethereum', 'tether']
        prices = get_crypto_prices(crypto_ids)
        print(prices)
        # Check that output is not None
        self.assertIsNotNone(prices)

        # Check that output has correct keys
        self.assertIn('bitcoin', prices)
        self.assertIn('ethereum', prices)
        self.assertIn('tether', prices)

        # Check that sub-dictionaries have 'usd' as a key
        for crypto in crypto_ids:
            self.assertIn('usd', prices[crypto])
            self.assertIsInstance(prices[crypto]['usd'], (int, float))

if __name__ == '__main__':
    unittest.main()
