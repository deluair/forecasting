# Quick Start Guide

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd forecasting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize project directories:
```bash
python init_project.py
```

## Basic Example

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

# Split
train, test = data.split(0.8)

# Fit and predict
forecaster = ARIMAForecaster(auto=True)
forecaster.fit(train)
forecast = forecaster.predict(horizon=len(test.values))

# Evaluate
metrics = MetricSuite()
results = metrics.evaluate(forecast, test)
print(results)
```

## Running Examples

```bash
# Basic forecasting
python examples/basic_forecasting.py

# Competition workflow
python examples/competition_workflow.py

# Advanced ensemble
python examples/advanced_ensemble.py
```

## Project Structure

- `src/`: Source code
- `examples/`: Example scripts
- `notebooks/`: Jupyter notebooks
- `tests/`: Unit tests
- `data/`: Data storage
- `config/`: Configuration files

## Next Steps

1. Read `docs/README.md` for detailed documentation
2. Explore `examples/` for more advanced usage
3. Check `notebooks/` for interactive examples
4. Review `config/default.yaml` for configuration options

## Getting Help

- Check documentation in `docs/`
- Review example scripts in `examples/`
- Run tests: `pytest tests/`

