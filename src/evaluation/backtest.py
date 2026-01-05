import numpy as np
import pandas as pd


class SimpleBacktest:
    """Backtest a simple trading strategy"""

    def __init__(self, df, predictions):
        self.df = df.copy()
        self.predictions = predictions

    def run_strategy(self, threshold=0.0):
        """
        Simple strategy:
        - Buy if predicted return > threshold
        - Sell if predicted return < -threshold
        - Hold otherwise
        """
        # Align predictions with data
        self.df = self.df.iloc[-len(self.predictions):].copy()
        self.df['predicted_return'] = self.predictions

        # Generate signals
        self.df['signal'] = 0
        self.df.loc[self.df['predicted_return'] > threshold, 'signal'] = 1  # Buy
        self.df.loc[self.df['predicted_return'] < -threshold, 'signal'] = -1  # Sell

        # Calculate strategy returns
        self.df['actual_return'] = self.df['Close'].pct_change()
        self.df['strategy_return'] = self.df['signal'].shift(1) * self.df['actual_return']

        # Calculate cumulative returns
        self.df['cumulative_market'] = (1 + self.df['actual_return']).cumprod()
        self.df['cumulative_strategy'] = (1 + self.df['strategy_return']).cumprod()

        # Performance metrics
        total_return_market = self.df['cumulative_market'].iloc[-1] - 1
        total_return_strategy = self.df['cumulative_strategy'].iloc[-1] - 1

        sharpe_ratio = self.df['strategy_return'].mean() / self.df['strategy_return'].std() * np.sqrt(252)

        print("\n" + "=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)
        print(f"Buy-and-Hold Return: {total_return_market:.2%}")
        print(f"Strategy Return:     {total_return_strategy:.2%}")
        print(f"Sharpe Ratio:        {sharpe_ratio:.2f}")