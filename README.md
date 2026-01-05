\# VFEX Stock Price Forecasting System



Short-term price movement and volatility forecasting for VFEX-listed stocks using machine learning.



\## Features



\- ✅ Data collection \& preprocessing

\- ✅ Technical indicator generation (RSI, MACD, Bollinger Bands)

\- ✅ Baseline models (Naive, ARIMA, Linear Regression)

\- ✅ ML models (Random Forest, XGBoost)

\- ✅ Comprehensive evaluation \& backtesting

\- ✅ Interactive dashboard



\## Installation

```bash

git clone <repo>

cd vfex\_forecasting

pip install -r requirements.txt

```



\## Quick Start

```bash

\# Run full pipeline

python main.py



\# Launch dashboard

streamlit run dashboard/app.py

```



\## Project Structure

````

vfex\_forecasting/

├── data/

│   ├── raw/              # Raw VFEX data

│   ├── processed/        # Cleaned \& engineered data

│   └── results/          # Model outputs

├── src/

│   ├── data/             # Data collection \& preprocessing

│   ├── features/         # Feature engineering

│   ├── models/           # Model implementations

│   └── evaluation/       # Evaluation \& backtesting

├── dashboard/            # Streamlit app

├── notebooks/            # Jupyter notebooks

└── tests/                # Unit tests

