import yfinance as yf
import pandas as pd


class MacroDataCollector:
    """Collect USD-based macro indicators"""

    def get_us_inflation(self):
        """CPI data as proxy"""
        # Use FRED or manual data
        pass

    def get_fed_rates(self):
        """Federal funds rate"""
        # Download from FRED
        pass

    def get_usd_index(self):
        """US Dollar Index"""
        dxy = yf.download('DX-Y.NYB', start='2020-01-01')
        return dxy['Close']

    def get_vix(self):
        """Volatility index"""
        vix = yf.download('^VIX', start='2020-01-01')
        return vix['Close']


# Collect and save
collector = MacroDataCollector()
dxy = collector.get_usd_index()
dxy.to_csv('data/raw/usd_index.csv')