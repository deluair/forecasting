# Forecasting Project Documentation

## Overview

This is an advanced forecasting framework designed for prediction competitions like Metaculus and GJ Open. The project provides comprehensive tools for building, evaluating, and visualizing forecasts.

## Project Structure

```
forecasting/
├── src/                    # Source code
│   ├── core/              # Core infrastructure
│   ├── models/            # Forecasting models
│   ├── data/              # Data handlers
│   ├── evaluation/        # Metrics and scoring
│   ├── visualization/     # Plotting tools
│   └── utils/             # Utilities
├── data/                   # Data storage
├── notebooks/              # Jupyter notebooks
├── examples/               # Example scripts
├── tests/                  # Unit tests
└── config/                 # Configuration files
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from src.core import ForecastData
from src.models import ARIMAForecaster
from src.evaluation import MetricSuite

# Create data
dates = pd.date_range('2020-01-01', periods=100, freq='D')
values = np.random.randn(100).cumsum()
data = ForecastData(dates, values)

# Split data
train, test = data.split(0.8)

# Fit model
forecaster = ARIMAForecaster(auto=True)
forecaster.fit(train)

# Predict
forecast = forecaster.predict(horizon=len(test.values))

# Evaluate
metrics = MetricSuite()
results = metrics.evaluate(forecast, test)
```

## Models

### Available Models

1. **ARIMAForecaster**: AutoRegressive Integrated Moving Average
2. **ProphetForecaster**: Facebook Prophet for time series with seasonality
3. **MLForecaster**: Machine learning models (Random Forest, Gradient Boosting, etc.)
4. **BayesianForecaster**: Bayesian probabilistic forecasting
5. **EnsembleForecaster**: Combines multiple models
6. **WeightedEnsemble**: Optimized weighted ensemble

### Model Selection

- **Time series with trend/seasonality**: Use ProphetForecaster
- **Auto-regressive patterns**: Use ARIMAForecaster
- **Complex patterns**: Use MLForecaster or EnsembleForecaster
- **Uncertainty quantification**: Use BayesianForecaster

## Evaluation Metrics

### Available Metrics

- **BrierScore**: For probabilistic forecasts (lower is better)
- **LogScore**: Logarithmic scoring rule (higher is better)
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **CalibrationScore**: Measures forecast calibration
- **SharpnessScore**: Measures prediction concentration
- **CRPS**: Continuous Ranked Probability Score

### Usage

```python
from src.evaluation import BrierScore, MetricSuite

# Single metric
brier = BrierScore()
score = brier.evaluate(forecast, actuals)

# Multiple metrics
metrics = MetricSuite()
results = metrics.evaluate(forecast, actuals)
```

## Competition Data

### Metaculus

```python
from src.data import MetaculusLoader

loader = MetaculusLoader("data/metaculus_questions.csv")
df = loader.load()
questions = loader.parse_questions(df)
```

### GJ Open

```python
from src.data import GJOpenLoader

loader = GJOpenLoader("data/gj_open_questions.csv")
df = loader.load()
questions = loader.parse_questions(df)
```

## Visualization

```python
from src.visualization import ForecastPlotter

plotter = ForecastPlotter()
fig = plotter.plot_forecast(
    forecast,
    actuals=test_data,
    train_data=train_data
)
```

## Calibration

```python
from src.utils import CalibrationTool

calibrated, calibrate_func = CalibrationTool.isotonic_calibration(
    predictions, actuals
)
```

## Examples

See the `examples/` directory for:
- `basic_forecasting.py`: Basic workflow
- `competition_workflow.py`: Competition-specific workflow
- `advanced_ensemble.py`: Advanced ensemble methods

## Best Practices

1. **Data Splitting**: Always use proper train/validation/test splits
2. **Cross-Validation**: Use time-series cross-validation for evaluation
3. **Ensemble**: Combine multiple models for better performance
4. **Calibration**: Calibrate probabilistic forecasts
5. **Uncertainty**: Always quantify and report uncertainty
6. **Evaluation**: Use multiple metrics, not just one

## Contributing

1. Follow PEP 8 style guidelines
2. Add tests for new features
3. Update documentation
4. Use type hints

## License

MIT License

