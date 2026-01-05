import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


class FeatureEngineer:
    """Generate technical indicators and features"""

    def __init__(self, df):
        self.df = df.copy()
        self.df = self.df.sort_values('Date').reset_index(drop=True)

    def create_all_features(self):
        """Generate all features"""
        self._create_returns()
        self._create_moving_averages()
        self._create_momentum_indicators()
        self._create_volatility_indicators()
        self._create_volume_indicators()
        self._create_lag_features()
        self._create_temporal_features()

        # Drop rows with NaN (from indicator calculations)
        self.df = self.df.dropna()

        return self.df

    def _create_returns(self):
        """Log returns and simple returns"""
        self.df['log_return'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        self.df['simple_return'] = self.df['Close'].pct_change()

        # Multi-period returns
        for period in [5, 10, 20]:
            self.df[f'return_{period}d'] = self.df['Close'].pct_change(period)

    def _create_moving_averages(self):
        """SMA and EMA"""
        for window in [5, 10, 20, 50]:
            # Simple Moving Average
            sma = SMAIndicator(close=self.df['Close'], window=window)
            self.df[f'sma_{window}'] = sma.sma_indicator()

            # Exponential Moving Average
            ema = EMAIndicator(close=self.df['Close'], window=window)
            self.df[f'ema_{window}'] = ema.ema_indicator()

            # Price relative to MA
            self.df[f'price_to_sma_{window}'] = self.df['Close'] / self.df[f'sma_{window}']

    def _create_momentum_indicators(self):
        """RSI, MACD"""
        # RSI
        rsi = RSIIndicator(close=self.df['Close'], window=14)
        self.df['rsi'] = rsi.rsi()

        # MACD
        macd = MACD(close=self.df['Close'])
        self.df['macd'] = macd.macd()
        self.df['macd_signal'] = macd.macd_signal()
        self.df['macd_diff'] = macd.macd_diff()

    def _create_volatility_indicators(self):
        """Rolling volatility, Bollinger Bands"""
        # Rolling standard deviation
        for window in [5, 10, 20]:
            self.df[f'volatility_{window}d'] = self.df['log_return'].rolling(window).std()

        # Bollinger Bands
        bb = BollingerBands(close=self.df['Close'], window=20, window_dev=2)
        self.df['bb_high'] = bb.bollinger_hband()
        self.df['bb_low'] = bb.bollinger_lband()
        self.df['bb_width'] = (self.df['bb_high'] - self.df['bb_low']) / self.df['Close']

    def _create_volume_indicators(self):
        """Volume-based features"""
        # On-Balance Volume (OBV)
        self.df['obv'] = (np.sign(self.df['Close'].diff()) * self.df['Volume']).fillna(0).cumsum()

        # Volume moving average
        self.df['volume_sma_20'] = self.df['Volume'].rolling(20).mean()
        self.df['volume_ratio'] = self.df['Volume'] / self.df['volume_sma_20']

    def _create_lag_features(self):
        """Lagged values"""
        for lag in [1, 2, 3, 5]:
            self.df[f'close_lag_{lag}'] = self.df['Close'].shift(lag)
            self.df[f'volume_lag_{lag}'] = self.df['Volume'].shift(lag)
            self.df[f'return_lag_{lag}'] = self.df['simple_return'].shift(lag)

    def _create_temporal_features(self):
        """Date-based features"""
        self.df['day_of_week'] = self.df['Date'].dt.dayofweek
        self.df['month'] = self.df['Date'].dt.month
        self.df['quarter'] = self.df['Date'].dt.quarter


# Usage
df = pd.read_csv('data/processed/SEED_VFEX_clean.csv')
engineer = FeatureEngineer(df)
df_features = engineer.create_all_features()
df_features.to_csv('data/processed/SEED_VFEX_features.csv', index=False)