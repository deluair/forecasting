# API Documentation

Complete API reference for the Advanced Forecasting Framework.

## Core Module (`src.core`)

### `ForecastData`

Container for time series data with metadata.

```python
class ForecastData:
    def __init__(
        self,
        timestamps: Union[List, np.ndarray, pd.DatetimeIndex],
        values: Union[List, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    )
```

**Parameters:**
- `timestamps`: Time indices for observations
- `values`: Observed values
- `metadata`: Optional dictionary with additional information

**Methods:**
- `split(split_point)`: Split data into train/test sets
- `to_dataframe()`: Convert to pandas DataFrame

### `ForecastResult`

Container for forecast results with uncertainty quantification.

```python
class ForecastResult:
    def __init__(
        self,
        point_forecast: Union[List, np.ndarray],
        timestamps: Optional[Union[List, np.ndarray, pd.DatetimeIndex]] = None,
        lower_bound: Optional[Union[List, np.ndarray]] = None,
        upper_bound: Optional[Union[List, np.ndarray]] = None,
        quantiles: Optional[Dict[float, Union[List, np.ndarray]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    )
```

**Methods:**
- `get_uncertainty_intervals(confidence=0.95)`: Extract uncertainty intervals
- `to_dataframe()`: Convert to pandas DataFrame

### `BaseForecaster`

Abstract base class for all forecasters.

```python
class BaseForecaster(ABC):
    @abstractmethod
    def fit(self, data: ForecastData, **kwargs) -> 'BaseForecaster'
    
    @abstractmethod
    def predict(
        self,
        horizon: Optional[int] = None,
        timestamps: Optional[Union[List, np.ndarray, pd.DatetimeIndex]] = None,
        return_quantiles: Optional[List[float]] = None,
        **kwargs
    ) -> ForecastResult
```

## Models (`src.models`)

### `ARIMAForecaster`

AutoRegressive Integrated Moving Average forecaster.

```python
class ARIMAForecaster(BaseForecaster):
    def __init__(
        self,
        order: Optional[tuple] = None,
        seasonal_order: Optional[tuple] = None,
        auto: bool = True,
        name: Optional[str] = None
    )
```

**Parameters:**
- `order`: (p, d, q) tuple for ARIMA order
- `seasonal_order`: (P, D, Q, s) for seasonal component
- `auto`: If True, automatically select order using AIC/BIC

### `ProphetForecaster`

Facebook Prophet for time series with seasonality.

```python
class ProphetForecaster(BaseForecaster):
    def __init__(
        self,
        growth: str = 'linear',
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        seasonality_mode: str = 'additive',
        name: Optional[str] = None,
        **prophet_kwargs
    )
```

### `EnsembleForecaster`

Combine multiple forecasters.

```python
class EnsembleForecaster(BaseForecaster):
    def __init__(
        self,
        forecasters: Optional[List[BaseForecaster]] = None,
        weights: Optional[Union[List[float], str]] = None,
        method: str = 'weighted_average',
        name: Optional[str] = None
    )
```

**Methods:**
- `add_forecaster(forecaster)`: Add a forecaster to ensemble
- `fit(data, validation_split=0.2)`: Fit all forecasters

### `MLForecaster`

Machine learning based forecaster.

```python
class MLForecaster(BaseForecaster):
    def __init__(
        self,
        model_type: str = 'random_forest',
        model_params: Optional[Dict[str, Any]] = None,
        n_lags: int = 10,
        name: Optional[str] = None
    )
```

**Model Types:**
- `'random_forest'`: Random Forest Regressor
- `'gradient_boosting'`: Gradient Boosting Regressor
- `'ridge'`: Ridge Regression
- `'lasso'`: Lasso Regression

## Evaluation (`src.evaluation`)

### `BrierScore`

Brier Score for probabilistic forecasts.

```python
class BrierScore(BaseMetric):
    def evaluate(
        self,
        predictions: Union[ForecastResult, np.ndarray],
        actuals: Union[ForecastData, np.ndarray],
        **kwargs
    ) -> float
```

### `MetricSuite`

Comprehensive evaluation suite.

```python
class MetricSuite:
    def __init__(self, metrics: Optional[List[BaseMetric]] = None)
    
    def evaluate(
        self,
        predictions: Union[ForecastResult, np.ndarray],
        actuals: Union[ForecastData, np.ndarray]
    ) -> Dict[str, float]
```

## Data Handlers (`src.data`)

### `MetaculusLoader`

Load Metaculus competition data.

```python
class MetaculusLoader(CompetitionDataLoader):
    def load(self) -> pd.DataFrame
    def parse_questions(self, df: pd.DataFrame) -> List[Dict]
    def get_binary_questions(self, df: pd.DataFrame) -> pd.DataFrame
    def get_continuous_questions(self, df: pd.DataFrame) -> pd.DataFrame
```

### `GJOpenLoader`

Load GJ Open competition data.

```python
class GJOpenLoader(CompetitionDataLoader):
    def load(self) -> pd.DataFrame
    def parse_questions(self, df: pd.DataFrame) -> List[Dict]
```

## Visualization (`src.visualization`)

### `ForecastPlotter`

Plotting utilities for forecasts.

```python
class ForecastPlotter:
    def plot_forecast(
        self,
        forecast: ForecastResult,
        actuals: Optional[ForecastData] = None,
        train_data: Optional[ForecastData] = None,
        show_uncertainty: bool = True,
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure
    
    def plot_residuals(
        self,
        forecast: ForecastResult,
        actuals: ForecastData,
        title: Optional[str] = None,
        figsize: tuple = (12, 4)
    ) -> plt.Figure
    
    def plot_calibration(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        n_bins: int = 10,
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure
```

## Utilities (`src.utils`)

### `CalibrationTool`

Calibration methods for probabilistic forecasts.

```python
class CalibrationTool:
    @staticmethod
    def isotonic_calibration(
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> Tuple[np.ndarray, callable]
    
    @staticmethod
    def platt_scaling(
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> Tuple[np.ndarray, callable]
```

### `UncertaintyQuantifier`

Uncertainty quantification methods.

```python
class UncertaintyQuantifier:
    @staticmethod
    def bootstrap_uncertainty(
        forecaster,
        data: ForecastData,
        n_samples: int = 100,
        horizon: int = 1,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
    
    @staticmethod
    def conformal_prediction(
        forecaster,
        train_data: ForecastData,
        calibration_data: ForecastData,
        horizon: int = 1,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

---

For detailed examples, see the `examples/` directory.

