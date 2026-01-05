import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


class BaselineModels:
    """Baseline forecasting models"""

    def __init__(self, df):
        self.df = df.copy()
        self.results = {}

    def naive_random_walk(self, test_size=60):
        """Random walk: tomorrow's price = today's price"""
        train = self.df.iloc[:-test_size]
        test = self.df.iloc[-test_size:]

        # Prediction: last known price
        predictions = [train['Close'].iloc[-1]] * len(test)

        mae = mean_absolute_error(test['Close'], predictions)
        rmse = np.sqrt(mean_squared_error(test['Close'], predictions))

        self.results['naive'] = {
            'predictions': predictions,
            'mae': mae,
            'rmse': rmse
        }

        print(f"Naive Random Walk - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        return predictions

    def arima_model(self, test_size=60, order=(1, 1, 1)):
        """ARIMA model"""
        train = self.df.iloc[:-test_size]['Close']
        test = self.df.iloc[-test_size:]['Close']

        # Fit ARIMA
        model = ARIMA(train, order=order)
        fitted = model.fit()

        # Forecast
        predictions = fitted.forecast(steps=test_size)

        mae = mean_absolute_error(test, predictions)
        rmse = np.sqrt(mean_squared_error(test, predictions))

        self.results['arima'] = {
            'predictions': predictions,
            'mae': mae,
            'rmse': rmse
        }

        print(f"ARIMA{order} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        return predictions

    def linear_regression_returns(self, test_size=60):
        """Linear regression on lagged returns"""
        # Prepare features
        df = self.df[['Close', 'Volume']].copy()
        df['return'] = df['Close'].pct_change()
        df['return_lag1'] = df['return'].shift(1)
        df['return_lag2'] = df['return'].shift(2)
        df['volume_change'] = df['Volume'].pct_change()
        df = df.dropna()

        # Split
        train = df.iloc[:-test_size]
        test = df.iloc[-test_size:]

        X_train = train[['return_lag1', 'return_lag2', 'volume_change']]
        y_train = train['return']
        X_test = test[['return_lag1', 'return_lag2', 'volume_change']]
        y_test = test['return']

        # Train
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict returns, convert to prices
        predicted_returns = model.predict(X_test)

        mae = mean_absolute_error(y_test, predicted_returns)
        rmse = np.sqrt(mean_squared_error(y_test, predicted_returns))

        self.results['linear_reg'] = {
            'predictions': predicted_returns,
            'mae': mae,
            'rmse': rmse
        }

        print(f"Linear Regression - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        return predicted_returns


# Usage
df = pd.read_csv('data/processed/SEED_VFEX_features.csv')
baselines = BaselineModels(df)
baselines.naive_random_walk()
baselines.arima_model()
baselines.linear_regression_returns()