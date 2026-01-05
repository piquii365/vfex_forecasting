import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


class ModelEvaluator:
    """Comprehensive model evaluation"""

    def __init__(self, actual, predictions, dates):
        self.actual = np.array(actual)
        self.predictions = np.array(predictions)
        self.dates = dates

    def evaluate_all(self):
        """Run all evaluation metrics"""
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)

        self._regression_metrics()
        self._directional_accuracy()
        self._plot_predictions()
        self._error_analysis()

    def _regression_metrics(self):
        """MAE, RMSE, MAPE"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        mae = mean_absolute_error(self.actual, self.predictions)
        rmse = np.sqrt(mean_squared_error(self.actual, self.predictions))
        mape = np.mean(np.abs((self.actual - self.predictions) / self.actual)) * 100

        print("\nRegression Metrics:")
        print(f"  MAE:  {mae:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAPE: {mape:.2f}%")

    def _directional_accuracy(self):
        """Confusion matrix for up/down prediction"""
        actual_direction = np.sign(self.actual)
        pred_direction = np.sign(self.predictions)

        # Map to labels
        actual_labels = ['Down' if x < 0 else 'Up' for x in actual_direction]
        pred_labels = ['Down' if x < 0 else 'Up' for x in pred_direction]

        print("\nDirectional Accuracy:")
        print(classification_report(actual_labels, pred_labels))

        # Confusion matrix
        cm = confusion_matrix(actual_labels, pred_labels, labels=['Down', 'Up'])

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Down', 'Up'],
                    yticklabels=['Down', 'Up'])
        plt.title('Directional Prediction Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('data/results/confusion_matrix.png', dpi=300)
        plt.close()

    def _plot_predictions(self):
        """Plot actual vs predicted"""
        plt.figure(figsize=(14, 6))
        plt.plot(self.dates, self.actual, label='Actual', color='black', linewidth=2)
        plt.plot(self.dates, self.predictions, label='Predicted', color='red', linewidth=1.5, alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.title('Actual vs Predicted Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('data/results/predictions_plot.png', dpi=300)
        plt.close()

    def _error_analysis(self):
        """Analyze prediction errors"""
        errors = self.actual - self.predictions

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.hist(errors, bins=30, edgecolor='black')
        plt.axvline(0, color='red', linestyle='--')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')

        plt.subplot(1, 2, 2)
        plt.scatter(self.predictions, errors, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Predicted Value')
        plt.ylabel('Error')
        plt.title('Residual Plot')

        plt.tight_layout()
        plt.savefig('data/results/error_analysis.png', dpi=300)
        plt.close()

# Usage
# After training model:
# evaluator = ModelEvaluator(y_test, predictions, test_dates)
# evaluator.evaluate_all()