# Advanced Forecasting for Prediction Competitions

A comprehensive, production-ready forecasting framework designed for prediction competitions like Metaculus and GJ Open. This project provides advanced forecasting models, evaluation metrics, calibration tools, and visualization capabilities.

## Features

- **Multiple Forecasting Models**: Ensemble methods, Bayesian approaches, time series models, and more
- **Prediction Competition Support**: Built-in handlers for Metaculus and GJ Open formats
- **Advanced Evaluation**: Brier scores, log scores, calibration metrics, and proper scoring rules
- **Uncertainty Quantification**: Probabilistic forecasts with confidence intervals
- **Calibration Tools**: Methods to improve forecast calibration
- **Visualization**: Comprehensive plotting and reporting tools
- **Modular Architecture**: Easy to extend with custom models and metrics

## Project Structure

```
forecasting/
├── src/
│   ├── core/              # Core forecasting infrastructure
│   ├── models/            # Forecasting models
│   ├── data/              # Data handlers and loaders
│   ├── evaluation/        # Metrics and scoring
│   ├── visualization/     # Plotting and reporting
│   └── utils/             # Utilities and helpers
├── data/                  # Data storage
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Unit tests
├── examples/              # Example scripts
└── config/                # Configuration files
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.models.ensemble import EnsembleForecaster
from src.data.competition_loader import MetaculusLoader
from src.evaluation.metrics import BrierScore

# Load competition data
loader = MetaculusLoader("data/metaculus_questions.csv")
data = loader.load()

# Train ensemble forecaster
forecaster = EnsembleForecaster()
forecaster.fit(data.train)

# Make predictions
predictions = forecaster.predict(data.test)

# Evaluate
brier = BrierScore()
score = brier.evaluate(predictions, data.test.actuals)
print(f"Brier Score: {score}")
```

## Documentation

See `docs/` for detailed documentation on:
- Model architecture
- API reference
- Competition format specifications
- Best practices

## License

MIT License

