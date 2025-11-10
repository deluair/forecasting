"""
Core forecasting infrastructure and base classes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime
import warnings


class ForecastData:
    """Container for forecast data with metadata."""
    
    def __init__(
        self,
        timestamps: Union[List, np.ndarray, pd.DatetimeIndex],
        values: Union[List, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize forecast data container.
        
        Parameters
        ----------
        timestamps : array-like
            Timestamps for the data points
        values : array-like
            Values corresponding to each timestamp
        metadata : dict, optional
            Additional metadata about the data
        """
        self.timestamps = pd.to_datetime(timestamps)
        self.values = np.asarray(values)
        self.metadata = metadata or {}
        
        if len(self.timestamps) != len(self.values):
            raise ValueError("Timestamps and values must have the same length")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame({
            'timestamp': self.timestamps,
            'value': self.values
        })
    
    def split(self, split_point: Union[int, float, datetime]) -> Tuple['ForecastData', 'ForecastData']:
        """
        Split data into train and test sets.
        
        Parameters
        ----------
        split_point : int, float, or datetime
            If int: index position
            If float: fraction of data (0-1)
            If datetime: timestamp
        
        Returns
        -------
        train_data : ForecastData
            Training data
        test_data : ForecastData
            Test data
        """
        if isinstance(split_point, float):
            split_idx = int(len(self.timestamps) * split_point)
        elif isinstance(split_point, datetime):
            split_idx = np.where(self.timestamps >= split_point)[0]
            if len(split_idx) == 0:
                split_idx = len(self.timestamps)
            else:
                split_idx = split_idx[0]
        else:
            split_idx = split_point
        
        train = ForecastData(
            self.timestamps[:split_idx],
            self.values[:split_idx],
            self.metadata
        )
        test = ForecastData(
            self.timestamps[split_idx:],
            self.values[split_idx:],
            self.metadata
        )
        
        return train, test


class ForecastResult:
    """Container for forecast results with uncertainty quantification."""
    
    def __init__(
        self,
        point_forecast: Union[List, np.ndarray],
        timestamps: Optional[Union[List, np.ndarray, pd.DatetimeIndex]] = None,
        lower_bound: Optional[Union[List, np.ndarray]] = None,
        upper_bound: Optional[Union[List, np.ndarray]] = None,
        quantiles: Optional[Dict[float, Union[List, np.ndarray]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize forecast result container.
        
        Parameters
        ----------
        point_forecast : array-like
            Point predictions
        timestamps : array-like, optional
            Timestamps for predictions
        lower_bound : array-like, optional
            Lower confidence bound
        upper_bound : array-like, optional
            Upper confidence bound
        quantiles : dict, optional
            Dictionary mapping quantile levels to predictions
        metadata : dict, optional
            Additional metadata about the forecast
        """
        self.point_forecast = np.asarray(point_forecast)
        self.timestamps = pd.to_datetime(timestamps) if timestamps is not None else None
        self.lower_bound = np.asarray(lower_bound) if lower_bound is not None else None
        self.upper_bound = np.asarray(upper_bound) if upper_bound is not None else None
        self.quantiles = quantiles or {}
        self.metadata = metadata or {}
        
        # Validate lengths
        n = len(self.point_forecast)
        if self.timestamps is not None and len(self.timestamps) != n:
            raise ValueError("Timestamps must match point_forecast length")
        if self.lower_bound is not None and len(self.lower_bound) != n:
            raise ValueError("Lower bound must match point_forecast length")
        if self.upper_bound is not None and len(self.upper_bound) != n:
            raise ValueError("Upper bound must match point_forecast length")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        data = {'point_forecast': self.point_forecast}
        
        if self.timestamps is not None:
            data['timestamp'] = self.timestamps
        
        if self.lower_bound is not None:
            data['lower_bound'] = self.lower_bound
        
        if self.upper_bound is not None:
            data['upper_bound'] = self.upper_bound
        
        for q, values in self.quantiles.items():
            data[f'quantile_{q}'] = values
        
        return pd.DataFrame(data)
    
    def get_uncertainty_intervals(self, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get uncertainty intervals from quantiles.
        
        Parameters
        ----------
        confidence : float
            Confidence level (e.g., 0.95 for 95% interval)
        
        Returns
        -------
        lower : np.ndarray
            Lower bounds
        upper : np.ndarray
            Upper bounds
        """
        alpha = 1 - confidence
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2
        
        if lower_q in self.quantiles and upper_q in self.quantiles:
            return self.quantiles[lower_q], self.quantiles[upper_q]
        elif self.lower_bound is not None and self.upper_bound is not None:
            return self.lower_bound, self.upper_bound
        else:
            warnings.warn("Uncertainty intervals not available")
            return np.full_like(self.point_forecast, np.nan), np.full_like(self.point_forecast, np.nan)


class BaseForecaster(ABC):
    """Abstract base class for all forecasters."""
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize base forecaster.
        
        Parameters
        ----------
        name : str, optional
            Name of the forecaster
        """
        self.name = name or self.__class__.__name__
        self.is_fitted = False
        self.training_data = None
    
    @abstractmethod
    def fit(self, data: ForecastData, **kwargs) -> 'BaseForecaster':
        """
        Fit the forecaster to training data.
        
        Parameters
        ----------
        data : ForecastData
            Training data
        **kwargs
            Additional model-specific parameters
        
        Returns
        -------
        self : BaseForecaster
            Returns self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        horizon: Optional[int] = None,
        timestamps: Optional[Union[List, np.ndarray, pd.DatetimeIndex]] = None,
        return_quantiles: Optional[List[float]] = None,
        **kwargs
    ) -> ForecastResult:
        """
        Generate forecasts.
        
        Parameters
        ----------
        horizon : int, optional
            Number of steps ahead to forecast
        timestamps : array-like, optional
            Specific timestamps to forecast
        return_quantiles : list of float, optional
            Quantile levels to return (e.g., [0.05, 0.95])
        **kwargs
            Additional prediction parameters
        
        Returns
        -------
        ForecastResult
            Forecast results with point predictions and uncertainty
        """
        pass
    
    def fit_predict(
        self,
        train_data: ForecastData,
        horizon: Optional[int] = None,
        timestamps: Optional[Union[List, np.ndarray, pd.DatetimeIndex]] = None,
        **kwargs
    ) -> ForecastResult:
        """
        Fit and predict in one step.
        
        Parameters
        ----------
        train_data : ForecastData
            Training data
        horizon : int, optional
            Forecast horizon
        timestamps : array-like, optional
            Timestamps to forecast
        **kwargs
            Additional parameters
        
        Returns
        -------
        ForecastResult
            Forecast results
        """
        self.fit(train_data, **kwargs)
        return self.predict(horizon=horizon, timestamps=timestamps, **kwargs)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"
