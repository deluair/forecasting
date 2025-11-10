# Forecasting Example Notebook

This directory contains Jupyter notebooks demonstrating the forecasting framework.

## Creating a Notebook

To create a new notebook:

1. Open Jupyter: `jupyter notebook`
2. Create a new Python notebook
3. Import the framework:

```python
from src.core import ForecastData
from src.models import ARIMAForecaster, ProphetForecaster, EnsembleForecaster
from src.evaluation import MetricSuite
from src.visualization import ForecastPlotter
```

## Example Workflow

See `examples/basic_forecasting.py` for a complete example that can be adapted to notebooks.

