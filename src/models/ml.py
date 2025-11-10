"""
Machine learning forecasting models.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import warnings

from ..core import BaseForecaster, ForecastData, ForecastResult


class MLForecaster(BaseForecaster):
    """
    Machine learning forecaster using scikit-learn models.
    
    Supports various ML models including:
    - Random Forest
    - Gradient Boosting
    - Ridge Regression
    - Lasso Regression
    """
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        model_params: Optional[Dict[str, Any]] = None,
        n_lags: int = 10,
        name: Optional[str] = None
    ):
        """
        Initialize ML forecaster.
        
        Parameters
        ----------
        model_type : str
            Type of ML model: 'random_forest', 'gradient_boosting', 'ridge', 'lasso'
        model_params : dict, optional
            Parameters for the ML model
        n_lags : int
            Number of lagged features to use
        name : str, optional
            Name of the forecaster
        """
        super().__init__(name)
        self.model_type = model_type
        self.model_params = model_params or {}
        self.n_lags = n_lags
        self.model = None
        self.scaler = StandardScaler()
        self.use_scaling = True
    
    def _create_model(self):
        """Create the ML model based on model_type."""
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(**self.model_params)
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(**self.model_params)
        elif self.model_type == 'ridge':
            self.model = Ridge(**self.model_params)
        elif self.model_type == 'lasso':
            self.model = Lasso(**self.model_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _create_lagged_features(self, values: np.ndarray) -> tuple:
        """
        Create lagged features from time series.
        
        Parameters
        ----------
        values : np.ndarray
            Time series values
        
        Returns
        -------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        """
        n = len(values)
        if n < self.n_lags + 1:
            raise ValueError(f"Need at least {self.n_lags + 1} observations")
        
        X = []
        y = []
        
        for i in range(self.n_lags, n):
            X.append(values[i - self.n_lags:i])
            y.append(values[i])
        
        return np.array(X), np.array(y)
    
    def fit(self, data: ForecastData, **kwargs) -> 'MLForecaster':
        """Fit ML model."""
        # Create lagged features
        X, y = self._create_lagged_features(data.values)
        
        # Scale features if needed
        if self.use_scaling:
            X = self.scaler.fit_transform(X)
        
        # Create and fit model
        self._create_model()
        self.model.fit(X, y)
        
        self.training_data = data
        self.is_fitted = True
        return self
    
    def predict(
        self,
        horizon: Optional[int] = None,
        timestamps: Optional[Union[List, np.ndarray, pd.DatetimeIndex]] = None,
        return_quantiles: Optional[List[float]] = None,
        **kwargs
    ) -> ForecastResult:
        """Generate ML forecast."""
        if not self.is_fitted:
            raise ValueError("Forecaster must be fitted before prediction")
        
        if horizon is None and timestamps is None:
            horizon = 1
        
        if timestamps is not None:
            horizon = len(timestamps)
        
        # Generate forecasts recursively
        predictions = []
        last_values = self.training_data.values[-self.n_lags:].copy()
        
        for _ in range(horizon):
            # Prepare features
            X_pred = last_values.reshape(1, -1)
            
            if self.use_scaling:
                X_pred = self.scaler.transform(X_pred)
            
            # Predict
            pred = self.model.predict(X_pred)[0]
            predictions.append(pred)
            
            # Update lagged values
            last_values = np.append(last_values[1:], pred)
        
        point_forecast = np.array(predictions)
        
        # Generate timestamps if not provided
        if timestamps is None:
            last_timestamp = self.training_data.timestamps[-1]
            if isinstance(self.training_data.timestamps, pd.DatetimeIndex):
                freq = pd.infer_freq(self.training_data.timestamps)
                if freq is None:
                    freq = 'D'
                timestamps = pd.date_range(
                    start=last_timestamp,
                    periods=horizon + 1,
                    freq=freq
                )[1:]
            else:
                timestamps = None
        
        # Estimate uncertainty using bootstrap or model-specific methods
        lower_bound = None
        upper_bound = None
        quantiles = {}
        
        if hasattr(self.model, 'predict_proba') or self.model_type in ['random_forest', 'gradient_boosting']:
            # For tree-based models, use prediction intervals
            if self.model_type in ['random_forest', 'gradient_boosting']:
                # Get predictions from all trees
                all_preds = []
                for _ in range(100):  # Bootstrap samples
                    last_values_boot = self.training_data.values[-self.n_lags:].copy()
                    preds_boot = []
                    for _ in range(horizon):
                        X_pred = last_values_boot.reshape(1, -1)
                        if self.use_scaling:
                            X_pred = self.scaler.transform(X_pred)
                        pred = self.model.predict(X_pred)[0]
                        preds_boot.append(pred)
                        last_values_boot = np.append(last_values_boot[1:], pred)
                    all_preds.append(preds_boot)
                
                all_preds = np.array(all_preds)
                lower_bound = np.percentile(all_preds, 2.5, axis=0)
                upper_bound = np.percentile(all_preds, 97.5, axis=0)
                
                if return_quantiles:
                    for q in return_quantiles:
                        quantiles[q] = np.percentile(all_preds, q * 100, axis=0)
        
        return ForecastResult(
            point_forecast=point_forecast,
            timestamps=timestamps,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            quantiles=quantiles,
            metadata={'model_type': self.model_type, 'n_lags': self.n_lags}
        )

