import pandas as pd
from src.data.vfex_scraper import VFEXDataCollector
from src.data.preprocessor import VFEXDataPreprocessor
from src.features.engineer import FeatureEngineer
from src.models.baselines import BaselineModels
from src.models.ml_models import MLForecaster
from src.evaluation.evaluator import ModelEvaluator


def main():
    print("=" * 60)
    print("VFEX STOCK FORECASTING PIPELINE")
    print("=" * 60)

    # Step 1: Data Collection
    print("\n[1/6] Collecting data...")
    # (Assuming manual collection for VFEX)

    # Step 2: Preprocessing
    print("\n[2/6] Preprocessing data...")
    df = pd.read_csv('data/raw/SEED.VFEX.csv')
    preprocessor = VFEXDataPreprocessor(df)
    df_clean = preprocessor.clean()
    df_clean.to_csv('data/processed/SEED_VFEX_clean.csv', index=False)

    # Step 3: Feature Engineering
    print("\n[3/6] Engineering features...")
    engineer = FeatureEngineer(df_clean)
    df_features = engineer.create_all_features()
    df_features.to_csv('data/processed/SEED_VFEX_features.csv', index=False)

    # Step 4: Baseline Models
    print("\n[4/6] Training baseline models...")
    baselines = BaselineModels(df_features)
    baselines.naive_random_walk()
    baselines.arima_model()

    # Step 5: ML Models
    print("\n[5/6] Training ML models...")
    forecaster = MLForecaster(df_features)
    rf_model, rf_preds = forecaster.train_random_forest()
    xgb_model, xgb_preds = forecaster.train_xgboost()
    forecaster.save_models()

    # Step 6: Evaluation
    print("\n[6/6] Evaluating models...")
    # (Add evaluation code here)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
