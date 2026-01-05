import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


class MLForecaster:
    """Machine learning forecasting models"""

    def __init__(self, df):
        self.df = df.copy()
        self.models = {}
        self.results = {}

    def prepare_data(self, target='return_1d', test_size=60):
        """Prepare features and target"""
        # Create next-day return as target
        self.df['return_1d'] = self.df['Close'].pct_change().shift(-1)

        # Feature columns (exclude target and non-features)
        exclude_cols = ['Date', 'Close', 'return_1d', 'Open', 'High', 'Low']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]

        # Drop NaN
        df_clean = self.df.dropna()

        # Split
        train = df_clean.iloc[:-test_size]
        test = df_clean.iloc[-test_size:]

        X_train = train[feature_cols]
        y_train = train[target]
        X_test = test[feature_cols]
        y_test = test[target]

        return X_train, X_test, y_train, y_test, feature_cols

    def train_random_forest(self, test_size=60):
        """Train Random Forest"""
        X_train, X_test, y_train, y_test, features = self.prepare_data(test_size=test_size)

        # Train
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Evaluate
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        # Directional accuracy
        direction_actual = np.sign(y_test)
        direction_pred = np.sign(predictions)
        directional_accuracy = (direction_actual == direction_pred).mean()

        self.models['random_forest'] = model
        self.results['random_forest'] = {
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy,
            'feature_importance': dict(zip(features, model.feature_importances_))
        }

        print(f"\n=== Random Forest ===")
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"Directional Accuracy: {directional_accuracy:.2%}")

        # Top features
        importance = sorted(
            self.results['random_forest']['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        print("\nTop 10 Features:")
        for feat, imp in importance:
            print(f"  {feat}: {imp:.4f}")

        return model, predictions

    def train_xgboost(self, test_size=60):
        """Train XGBoost"""
        X_train, X_test, y_train, y_test, features = self.prepare_data(test_size=test_size)

        # Train
        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Evaluate
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        direction_actual = np.sign(y_test)
        direction_pred = np.sign(predictions)
        directional_accuracy = (direction_actual == direction_pred).mean()

        self.models['xgboost'] = model
        self.results['xgboost'] = {
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy
        }

        print(f"\n=== XGBoost ===")
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"Directional Accuracy: {directional_accuracy:.2%}")

        return model, predictions

    def save_models(self, path='models/'):
        """Save trained models"""
        for name, model in self.models.items():
            joblib.dump(model, f'{path}{name}.pkl')
        print(f"Models saved to {path}")


# Usage
df = pd.read_csv('data/processed/SEED_VFEX_features.csv')
forecaster = MLForecaster(df)
forecaster.train_random_forest()
forecaster.train_xgboost()
forecaster.save_models()