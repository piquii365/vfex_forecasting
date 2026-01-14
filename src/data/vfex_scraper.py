# python
from pathlib import Path
import pandas as pd
import requests
from datetime import datetime, timedelta
import time


class VFEXDataCollector:
    """
    Collects VFEX stock data.
    - Ensures data folder exists.
    - Creates a template CSV when a manual CSV is missing.
    """

    def __init__(self, data_dir: str = "data/raw"):
        self.base_url = "https://www.vfex.exchange/"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _parse_date(self, d):
        if isinstance(d, (datetime,)):
            return d
        return pd.to_datetime(d)

    def create_manual_csv_template(self, ticker, start_date, end_date):
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)
        if end < start:
            end = start
        dates = pd.date_range(start=start, end=end, freq="B")  # business days
        df = pd.DataFrame({
            "Date": dates,
            "Open": pd.NA,
            "High": pd.NA,
            "Low": pd.NA,
            "Close": pd.NA,
            "Volume": pd.NA,
        })
        path = self.data_dir / f"{ticker}_manual.csv"
        df.to_csv(path, index=False)
        print(f"Template created at `{path}` — please populate with actual OHLCV data.")
        return df

    def download_historical_data(self, ticker, start_date, end_date):
        """
        Attempts to read `data/raw/{ticker}_manual.csv`.
        If missing, creates a template CSV (so you can fill it) and returns that template.
        """
        path = self.data_dir / f"{ticker}_manual.csv"
        try:
            df = pd.read_csv(path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            return df
        except FileNotFoundError:
            print(f"Missing manual CSV for `{ticker}`. Creating template...")
            return self.create_manual_csv_template(ticker, start_date, end_date)
        except Exception as e:
            print(f"Error downloading `{ticker}`: {e}")
            return None

    def collect_all_stocks(self, start_date='2020-01-01'):
        """Collect data for all stocks"""
        stocks = self.get_stock_list()
        all_data = {}

        for ticker in stocks:
            print(f"Downloading {ticker}...")
            df = self.download_historical_data(
                ticker,
                start_date,
                datetime.now().strftime('%Y-%m-%d')
            )

            if df is not None:
                all_data[ticker] = df
                df.to_csv(self.data_dir / f'{ticker}.csv', index=False)

            time.sleep(1)  # Be respectful

        return all_data

    def get_stock_list(self):
        """Get list of all VFEX-listed stocks"""
        return [
            'SEED.VFEX',
            'PPCZ.VFEX',
            'ZBFH.VFEX',
        ]


# Usage (keep in file as before)
if __name__ == "__main__":
    collector = VFEXDataCollector()
    data = collector.collect_all_stocks()
    print(f"Collected data for {len(data)} stocks")