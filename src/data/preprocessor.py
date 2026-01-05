import pandas as pd
import numpy as np


class VFEXDataPreprocessor:
    """Clean and validate VFEX data"""

    def __init__(self, df):
        self.df = df.copy()

    def clean(self):
        """Full cleaning pipeline"""
        self._validate_columns()
        self._handle_missing_dates()
        self._remove_duplicates()
        self._handle_missing_values()
        self._detect_outliers()
        self._validate_ohlc()
        return self.df

    def _validate_columns(self):
        """Ensure required columns exist"""
        required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)

    def _handle_missing_dates(self):
        """Fill missing trading days"""
        # Create complete date range
        date_range = pd.date_range(
            start=self.df['Date'].min(),
            end=self.df['Date'].max(),
            freq='B'  # Business days
        )

        # Reindex to include all business days
        self.df = self.df.set_index('Date').reindex(date_range)
        self.df.index.name = 'Date'
        self.df = self.df.reset_index()

    def _handle_missing_values(self):
        """Forward fill prices, zero fill volume"""
        price_cols = ['Open', 'High', 'Low', 'Close']
        self.df[price_cols] = self.df[price_cols].ffill()
        self.df['Volume'] = self.df['Volume'].fillna(0)

    def _remove_duplicates(self):
        """Remove duplicate dates"""
        self.df = self.df.drop_duplicates(subset=['Date'], keep='last')

    def _detect_outliers(self):
        """Flag suspicious price movements"""
        self.df['Returns'] = self.df['Close'].pct_change()

        # Flag returns > 50% (likely errors)
        suspicious = np.abs(self.df['Returns']) > 0.5
        if suspicious.sum() > 0:
            print(f"⚠️ Warning: {suspicious.sum()} suspicious returns detected")
            print(self.df[suspicious][['Date', 'Close', 'Returns']])

    def _validate_ohlc(self):
        """Ensure High >= Low, Close between High/Low"""
        invalid = (self.df['High'] < self.df['Low']) | \
                  (self.df['Close'] > self.df['High']) | \
                  (self.df['Close'] < self.df['Low'])

        if invalid.sum() > 0:
            print(f"⚠️ Warning: {invalid.sum()} invalid OHLC rows")
            self.df = self.df[~invalid]


# Usage
df = pd.read_csv('data/raw/SEED.VFEX.csv')
preprocessor = VFEXDataPreprocessor(df)
clean_df = preprocessor.clean()
clean_df.to_csv('data/processed/SEED_VFEX_clean.csv', index=False)