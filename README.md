# Advanced Forecasting Framework for Prediction Competitions

A comprehensive, production-ready forecasting framework designed for prediction competitions like **Metaculus** and **GJ Open**. This project provides advanced forecasting models, evaluation metrics, calibration tools, and visualization capabilities suitable for academic research and professional economic forecasting.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎓 Academic Context

This framework implements state-of-the-art forecasting methodologies used in:
- **Economic forecasting** (GDP, inflation, exchange rates, financial markets)
- **Prediction markets** (Metaculus, GJ Open, Good Judgment Project)
- **Probabilistic forecasting** with proper scoring rules
- **Ensemble methods** for improved forecast accuracy
- **Calibration techniques** for well-calibrated probabilistic forecasts

### Theoretical Foundations

The framework is grounded in:
- **Proper Scoring Rules** (Gneiting & Raftery, 2007)
- **Forecast Combination** (Timmermann, 2006)
- **Calibration Methods** (Platt, 1999; Zadrozny & Elkan, 2002)
- **Time Series Econometrics** (Hamilton, 1994; Lütkepohl, 2005)
- **Bayesian Forecasting** (West & Harrison, 1997)

## 🌟 Key Features

### Forecasting Models
- **Time Series Models**: ARIMA, SARIMA, VAR, State Space Models
- **Machine Learning**: Random Forest, Gradient Boosting, Neural Networks
- **Bayesian Methods**: Gaussian Processes, Hierarchical Models
- **Ensemble Methods**: Weighted Averaging, Stacking, Bayesian Model Averaging
- **Econometric Models**: Vector Autoregression (VAR), Error Correction Models

### Evaluation & Metrics
- **Proper Scoring Rules**: Brier Score, Logarithmic Score, CRPS
- **Calibration Metrics**: Expected Calibration Error (ECE), Reliability Diagrams
- **Statistical Tests**: Diebold-Mariano, Ljung-Box, Augmented Dickey-Fuller
- **Information Criteria**: AIC, BIC, WAIC (for Bayesian models)

### Advanced Features
- **Uncertainty Quantification**: Bootstrap, Conformal Prediction, Bayesian Credible Intervals
- **Calibration Tools**: Isotonic Regression, Platt Scaling, Temperature Scaling
- **Feature Engineering**: Time-based features, lagged variables, rolling statistics
- **Visualization**: Forecast plots, calibration curves, residual analysis

## 📊 Project Structure

```
forecasting/
├── src/
│   ├── core/                  # Core infrastructure
│   │   ├── base.py           # BaseForecaster, ForecastData, ForecastResult
│   │   └── validation.py      # Statistical tests and validation
│   ├── models/                # Forecasting models
│   │   ├── ensemble.py       # Ensemble methods
│   │   ├── time_series.py    # ARIMA, Prophet
│   │   ├── econometric.py    # VAR, VECM, State Space (NEW)
│   │   ├── bayesian.py       # Bayesian models
│   │   └── ml.py             # Machine learning models
│   ├── data/                  # Data handlers
│   │   ├── competition_loader.py
│   │   └── economic_data.py  # Economic data loaders (NEW)
│   ├── evaluation/            # Metrics and scoring
│   │   ├── metrics.py        # Scoring rules
│   │   └── statistical_tests.py  # Statistical tests (NEW)
│   ├── visualization/         # Plotting tools
│   │   └── plotting.py
│   └── utils/                 # Utilities
│       ├── calibration.py    # Calibration methods
│       ├── uncertainty.py    # Uncertainty quantification
│       └── feature_engineering.py
├── examples/
│   ├── basic_forecasting.py
│   ├── competition_workflow.py
│   ├── advanced_ensemble.py
│   ├── bangladesh_economy_forecast.py
│   └── econometric_forecasting.py  # NEW: Advanced econometric example
├── docs/
│   ├── README.md
│   ├── API.md                 # API documentation (NEW)
│   ├── THEORY.md              # Theoretical background (NEW)
│   └── REFERENCES.md          # Academic references (NEW)
├── tests/                      # Comprehensive test suite
└── config/                     # Configuration files
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/deluair/forecasting.git
cd forecasting

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
from src.core import ForecastData
from src.models import ARIMAForecaster
from src.evaluation import MetricSuite
from src.evaluation.statistical_tests import DieboldMarianoTest
import pandas as pd
import numpy as np

# Create sample economic data
dates = pd.date_range('2020-01-01', periods=100, freq='M')
gdp_growth = np.random.randn(100).cumsum() * 0.1 + 2.5
data = ForecastData(dates, gdp_growth, metadata={'indicator': 'GDP Growth'})

# Split into train and test
train, test = data.split(0.8)

# Fit model
forecaster = ARIMAForecaster(auto=True)
forecaster.fit(train)

# Generate forecast with uncertainty
forecast = forecaster.predict(
    horizon=len(test.values),
    return_quantiles=[0.05, 0.25, 0.75, 0.95]
)

# Evaluate using proper scoring rules
metrics = MetricSuite()
results = metrics.evaluate(forecast, test)
print(f"Brier Score: {results['BrierScore']:.4f}")
print(f"Log Score: {results['LogScore']:.4f}")
print(f"CRPS: {results['CRPS']:.4f}")

# Statistical tests
dm_test = DieboldMarianoTest()
dm_statistic = dm_test.test(forecast.point_forecast, test.values)
print(f"Diebold-Mariano statistic: {dm_statistic:.4f}")
```

## 📈 Advanced Examples

### Economic Forecasting

```python
from src.models.econometric import VARForecaster
from src.data.economic_data import WorldBankLoader

# Load economic data
loader = WorldBankLoader()
data = loader.load_country_data('BGD', indicators=['NY.GDP.MKTP.KD.ZG', 'FP.CPI.TOTL.ZG'])

# Fit VAR model
var_model = VARForecaster(maxlags=4)
var_model.fit(data)

# Forecast with impulse response analysis
forecast = var_model.predict(horizon=8)
irf = var_model.impulse_response(periods=10)
```

### Ensemble Forecasting with Calibration

```python
from src.models import EnsembleForecaster, ARIMAForecaster, ProphetForecaster
from src.utils import CalibrationTool

# Create ensemble
ensemble = EnsembleForecaster(
    forecasters=[
        ARIMAForecaster(auto=True),
        ProphetForecaster(yearly_seasonality=True)
    ],
    method='stacking'
)
ensemble.fit(train_data)

# Generate probabilistic forecast
forecast = ensemble.predict(horizon=10, return_quantiles=[0.05, 0.95])

# Calibrate forecasts
calibrated, calibrate_func = CalibrationTool.isotonic_calibration(
    forecast.point_forecast,
    validation_data.values
)
```

## 🎯 Available Models

### Time Series Models

1. **ARIMAForecaster**: AutoRegressive Integrated Moving Average
   - Automatic order selection via AIC/BIC
   - Supports seasonal ARIMA (SARIMA)
   - Confidence intervals via asymptotic theory

2. **ProphetForecaster**: Facebook Prophet
   - Handles multiple seasonalities
   - Robust to missing data and outliers
   - Automatic changepoint detection

### Econometric Models

3. **VARForecaster**: Vector Autoregression
   - Multi-variable time series modeling
   - Impulse response functions
   - Forecast error variance decomposition

4. **VECMForecaster**: Vector Error Correction Model
   - Cointegration analysis
   - Long-run equilibrium relationships
   - Short-run dynamics

5. **StateSpaceForecaster**: State Space Models
   - Kalman filtering and smoothing
   - Structural time series models
   - Unobserved components

### Machine Learning Models

6. **MLForecaster**: Scikit-learn based models
   - Random Forest, Gradient Boosting
   - Ridge/Lasso regression
   - Feature engineering support

### Bayesian Models

7. **BayesianForecaster**: Probabilistic forecasting
   - Gaussian Process regression
   - Hierarchical models
   - Full posterior distributions

### Ensemble Methods

8. **EnsembleForecaster**: Model combination
   - Weighted averaging
   - Stacking (meta-learning)
   - Bayesian Model Averaging

9. **WeightedEnsemble**: Optimized weights
   - Inverse variance weighting
   - Cross-validation based optimization
   - Constrained optimization

## 📊 Evaluation Metrics

### Proper Scoring Rules

- **Brier Score**: For binary/probabilistic forecasts
- **Logarithmic Score**: Proper scoring rule for probabilities
- **CRPS**: Continuous Ranked Probability Score
- **Spherical Score**: Alternative proper scoring rule

### Calibration Metrics

- **Expected Calibration Error (ECE)**: Overall calibration measure
- **Reliability Diagrams**: Visual calibration assessment
- **Sharpness**: Measure of forecast concentration

### Statistical Tests

- **Diebold-Mariano Test**: Forecast accuracy comparison
- **Ljung-Box Test**: Residual autocorrelation
- **Augmented Dickey-Fuller**: Unit root testing
- **Jarque-Bera**: Normality test

## 🏆 Prediction Competitions

### Metaculus

```python
from src.data import MetaculusLoader

loader = MetaculusLoader("data/metaculus_questions.csv")
df = loader.load()

# Parse questions
questions = loader.parse_questions(df)

# Get binary questions for probability forecasting
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
from src.visualization import ForecastPlotter, ForecastReport

plotter = ForecastPlotter(style='seaborn-v0_8', figsize=(14, 8))

# Plot forecast with uncertainty
fig = plotter.plot_forecast(
    forecast,
    actuals=test_data,
    train_data=train_data,
    show_uncertainty=True,
    title='GDP Growth Forecast'
)

# Generate comprehensive report
report = ForecastReport()
report.generate_report(
    forecast=forecast,
    actuals=test_data,
    train_data=train_data,
    metrics=results,
    save_path='results/forecast_report.html'
)
```

## 🔧 Advanced Features

### Calibration

```python
from src.utils import CalibrationTool

# Isotonic regression (non-parametric)
calibrated, calibrate_func = CalibrationTool.isotonic_calibration(
    predictions, actuals
)

# Platt scaling (parametric)
calibrated, calibrate_func = CalibrationTool.platt_scaling(
    predictions, actuals
)

# Temperature scaling (for neural networks)
calibrated, temperature = CalibrationTool.temperature_scaling(
    logits, actuals
)
```

### Uncertainty Quantification

```python
from src.utils import UncertaintyQuantifier

# Bootstrap uncertainty
point, lower, upper = UncertaintyQuantifier.bootstrap_uncertainty(
    forecaster, data, n_samples=1000, confidence=0.95
)

# Conformal prediction
point, lower, upper = UncertaintyQuantifier.conformal_prediction(
    forecaster, train_data, calibration_data, confidence=0.95
)
```

## 📚 Documentation

- [Quick Start Guide](QUICKSTART.md)
- [API Documentation](docs/API.md)
- [Theoretical Background](docs/THEORY.md)
- [Academic References](docs/REFERENCES.md)
- [Examples](examples/)

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_econometric.py
```

## 🛠️ Requirements

See [requirements.txt](requirements.txt) for complete list. Key dependencies:

- **Core**: numpy, pandas, scipy
- **Time Series**: statsmodels, pmdarima, prophet
- **Econometrics**: statsmodels (VAR, VECM)
- **Bayesian**: pymc, arviz
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn, plotly

## 📖 Academic References

Key papers and books referenced in this framework:

1. Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359-378.

2. Timmermann, A. (2006). Forecast combinations. *Handbook of Economic Forecasting*, 1, 135-196.

3. Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263.

4. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.

5. Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.

See [docs/REFERENCES.md](docs/REFERENCES.md) for complete bibliography.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This framework is designed for:
- **Prediction Markets**: [Metaculus](https://www.metaculus.com/), [GJ Open](https://www.gjopen.com/)
- **Economic Forecasting**: Central banks, research institutions
- **Academic Research**: Forecasting competitions, econometric analysis

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/deluair/forecasting/issues)
- **Discussions**: [GitHub Discussions](https://github.com/deluair/forecasting/discussions)

## 🔗 Repository

**GitHub**: https://github.com/deluair/forecasting

---

**Status**: ✅ Production Ready | **Version**: 0.1.0 | **Last Updated**: 2024

*Designed for economists, researchers, and forecasting practitioners.*
