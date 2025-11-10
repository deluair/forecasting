# Advanced Forecasting for Prediction Competitions

A comprehensive, production-ready forecasting framework designed for prediction competitions like **Metaculus** and **GJ Open**. This project provides advanced forecasting models, evaluation metrics, calibration tools, and visualization capabilities.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🌟 Features

- **Multiple Forecasting Models**: Ensemble methods, Bayesian approaches, time series models, and machine learning
- **Prediction Competition Support**: Built-in handlers for Metaculus and GJ Open formats
- **Advanced Evaluation**: Brier scores, log scores, calibration metrics, and proper scoring rules
- **Uncertainty Quantification**: Probabilistic forecasts with confidence intervals
- **Calibration Tools**: Methods to improve forecast calibration (isotonic regression, Platt scaling)
- **Visualization**: Comprehensive plotting and reporting tools
- **Modular Architecture**: Easy to extend with custom models and metrics
- **Real-World Examples**: Includes Bangladesh economy forecasting demonstration

## 📊 Project Structure

```
forecasting/
├── src/                    # Source code
│   ├── core/              # Core forecasting infrastructure
│   │   ├── base.py        # BaseForecaster, ForecastData, ForecastResult
│   │   └── __init__.py
│   ├── models/            # Forecasting models
│   │   ├── ensemble.py    # EnsembleForecaster, WeightedEnsemble
│   │   ├── time_series.py # ARIMAForecaster, ProphetForecaster
│   │   ├── bayesian.py    # BayesianForecaster
│   │   ├── ml.py          # MLForecaster
│   │   └── __init__.py
│   ├── data/              # Data handlers
│   │   ├── competition_loader.py  # MetaculusLoader, GJOpenLoader
│   │   └── __init__.py
│   ├── evaluation/        # Metrics and scoring
│   │   ├── metrics.py     # BrierScore, LogScore, MAE, RMSE, etc.
│   │   └── __init__.py
│   ├── visualization/     # Plotting tools
│   │   ├── plotting.py    # ForecastPlotter, ForecastReport
│   │   └── __init__.py
│   └── utils/             # Utilities
│       ├── utils.py       # CalibrationTool, UncertaintyQuantifier
│       └── __init__.py
├── examples/              # Example scripts
│   ├── basic_forecasting.py
│   ├── competition_workflow.py
│   ├── advanced_ensemble.py
│   └── bangladesh_economy_forecast.py  # Real-world example
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── config/                # Configuration files
├── notebooks/             # Jupyter notebooks
└── results/               # Generated forecasts and visualizations
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/deluair/forecasting.git
cd forecasting

# Install dependencies
pip install -r requirements.txt

# Initialize project directories
python init_project.py
```

### Basic Usage

```python
from src.core import ForecastData
from src.models import ARIMAForecaster
from src.evaluation import MetricSuite
import pandas as pd
import numpy as np

# Create sample data
dates = pd.date_range('2020-01-01', periods=100, freq='D')
values = np.random.randn(100).cumsum()
data = ForecastData(dates, values)

# Split into train and test
train, test = data.split(0.8)

# Fit model
forecaster = ARIMAForecaster(auto=True)
forecaster.fit(train)

# Generate forecast
forecast = forecaster.predict(horizon=len(test.values))

# Evaluate
metrics = MetricSuite()
results = metrics.evaluate(forecast, test)
print(results)
```

## 📈 Example: Bangladesh Economy Forecast

The project includes a complete example forecasting key Bangladeshi economic indicators:

```bash
python examples/bangladesh_economy_forecast.py
```

This example forecasts:
- GDP Growth Rate (%)
- Inflation Rate (%)
- Exchange Rate (BDT/USD)
- Export Growth Rate (%)

Results are saved to `results/` directory with visualizations and summary statistics.

## 🎯 Available Models

### 1. **ARIMAForecaster**
AutoRegressive Integrated Moving Average model with automatic parameter selection.

```python
from src.models import ARIMAForecaster

forecaster = ARIMAForecaster(auto=True)
forecaster.fit(train_data)
forecast = forecaster.predict(horizon=10)
```

### 2. **ProphetForecaster**
Facebook Prophet for time series with seasonality.

```python
from src.models import ProphetForecaster

forecaster = ProphetForecaster(
    yearly_seasonality=True,
    weekly_seasonality=True
)
forecaster.fit(train_data)
forecast = forecaster.predict(horizon=10)
```

### 3. **MLForecaster**
Machine learning models (Random Forest, Gradient Boosting, Ridge, Lasso).

```python
from src.models import MLForecaster

forecaster = MLForecaster(
    model_type='random_forest',
    n_lags=10
)
forecaster.fit(train_data)
forecast = forecaster.predict(horizon=10)
```

### 4. **BayesianForecaster**
Probabilistic forecasting using PyMC.

```python
from src.models import BayesianForecaster

forecaster = BayesianForecaster(model_type='gaussian_process')
forecaster.fit(train_data)
forecast = forecaster.predict(horizon=10)
```

### 5. **EnsembleForecaster**
Combine multiple models with weighted averaging or stacking.

```python
from src.models import EnsembleForecaster, ARIMAForecaster, ProphetForecaster

ensemble = EnsembleForecaster(
    forecasters=[
        ARIMAForecaster(auto=True),
        ProphetForecaster()
    ],
    method='weighted_average',
    weights='learned'
)
ensemble.fit(train_data)
forecast = ensemble.predict(horizon=10)
```

## 📊 Evaluation Metrics

The framework includes comprehensive evaluation metrics:

- **BrierScore**: For probabilistic forecasts (lower is better)
- **LogScore**: Logarithmic scoring rule (higher is better)
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **CalibrationScore**: Measures forecast calibration (ECE)
- **SharpnessScore**: Measures prediction concentration
- **CRPS**: Continuous Ranked Probability Score

```python
from src.evaluation import MetricSuite, BrierScore

# Single metric
brier = BrierScore()
score = brier.evaluate(forecast, actuals)

# Multiple metrics
metrics = MetricSuite()
results = metrics.evaluate(forecast, actuals)
```

## 🏆 Prediction Competitions

### Metaculus

```python
from src.data import MetaculusLoader

loader = MetaculusLoader("data/metaculus_questions.csv")
df = loader.load()
questions = loader.parse_questions(df)

# Get binary questions
binary_questions = loader.get_binary_questions(df)
```

### GJ Open

```python
from src.data import GJOpenLoader

loader = GJOpenLoader("data/gj_open_questions.csv")
df = loader.load()
questions = loader.parse_questions(df)
```

## 📈 Visualization

```python
from src.visualization import ForecastPlotter

plotter = ForecastPlotter()
fig = plotter.plot_forecast(
    forecast,
    actuals=test_data,
    train_data=train_data,
    title='Economic Forecast'
)
```

## 🔧 Calibration

Improve forecast calibration using isotonic regression or Platt scaling:

```python
from src.utils import CalibrationTool
import numpy as np

# Isotonic calibration
calibrated, calibrate_func = CalibrationTool.isotonic_calibration(
    predictions, actuals
)

# Platt scaling
calibrated, calibrate_func = CalibrationTool.platt_scaling(
    predictions, actuals
)
```

## 📚 Examples

The `examples/` directory contains:

1. **basic_forecasting.py**: Simple forecasting workflow
2. **competition_workflow.py**: Prediction competition pipeline
3. **advanced_ensemble.py**: Advanced ensemble methods with calibration
4. **bangladesh_economy_forecast.py**: Real-world economic forecasting

Run examples:
```bash
python examples/basic_forecasting.py
python examples/bangladesh_economy_forecast.py
```

## 🧪 Testing

```bash
pytest tests/
```

## 📖 Documentation

- [Quick Start Guide](QUICKSTART.md)
- [Detailed Documentation](docs/README.md)
- [Project Status](PROJECT_STATUS.md)

## 🛠️ Requirements

See [requirements.txt](requirements.txt) for full list. Key dependencies:

- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- statsmodels >= 0.14.0
- prophet >= 1.1.4
- matplotlib >= 3.7.0

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

MIT License

## 🙏 Acknowledgments

Designed for prediction competitions like:
- [Metaculus](https://www.metaculus.com/)
- [GJ Open](https://www.gjopen.com/)

## 📧 Contact

For questions or issues, please open an issue on [GitHub](https://github.com/deluair/forecasting/issues).

---

**Repository**: https://github.com/deluair/forecasting

**Status**: ✅ Production Ready
