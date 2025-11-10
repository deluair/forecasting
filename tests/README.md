# Tests

This directory contains the test suite for the forecasting framework.

## Running Tests

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_basic.py -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v
```

Run only unit tests:
```bash
pytest tests/ -m unit -v
```

Run only integration tests:
```bash
pytest tests/ -m integration -v
```

Skip slow tests:
```bash
pytest tests/ -m "not slow" -v
```

## Test Structure

Tests are organized by module:
- `test_basic.py` - Basic functionality tests
- `test_core.py` - Core infrastructure tests
- `test_models.py` - Forecasting model tests
- `test_evaluation.py` - Metrics and evaluation tests
- `test_utils.py` - Utility function tests

## Writing Tests

Follow these guidelines when writing tests:

1. Use descriptive test names (e.g., `test_arima_forecast_returns_correct_shape`)
2. Use pytest fixtures for common setup
3. Mark tests appropriately:
   - `@pytest.mark.unit` for unit tests
   - `@pytest.mark.integration` for integration tests
   - `@pytest.mark.slow` for slow-running tests
4. Aim for high code coverage (>80%)
5. Test edge cases and error conditions

## Example Test

```python
import pytest
from src.core import ForecastData
from src.models import ARIMAForecaster


@pytest.mark.unit
def test_arima_forecaster_fit():
    """Test that ARIMA forecaster can be fitted."""
    data = ForecastData(
        timestamps=pd.date_range('2020-01-01', periods=100),
        values=np.random.randn(100)
    )
    forecaster = ARIMAForecaster()
    forecaster.fit(data)
    assert forecaster.is_fitted
```

## Continuous Integration

Tests are automatically run on GitHub Actions for:
- Python 3.9, 3.10, 3.11
- Ubuntu, macOS, Windows

See `.github/workflows/ci.yml` for the full CI configuration.
