"""
Ensemble forecasting models.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Callable
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings

from ..core import BaseForecaster, ForecastData, ForecastResult


class EnsembleForecaster(BaseForecaster):
    """
    Ensemble forecaster that combines multiple forecasting models.
    
    Supports weighted averaging, stacking, and voting strategies.
    """
    
    def __init__(
        self,
        forecasters: Optional[List[BaseForecaster]] = None,
        weights: Optional[Union[List[float], str]] = None,
        method: str = 'weighted_average',
        name: Optional[str] = None
    ):
        """
        Initialize ensemble forecaster.
        
        Parameters
        ----------
        forecasters : list of BaseForecaster, optional
            List of forecasters to ensemble
        weights : list of float or str, optional
            Weights for each forecaster. If 'equal', uses equal weights.
            If 'learned', learns weights from validation data.
        method : str
            Ensemble method: 'weighted_average', 'stacking', or 'median'
        name : str, optional
            Name of the forecaster
        """
        super().__init__(name)
        self.forecasters = forecasters or []
        self.weights = weights
        self.method = method
        self.meta_model = None  # For stacking
        self.is_fitted = False
    
    def add_forecaster(self, forecaster: BaseForecaster) -> 'EnsembleForecaster':
        """Add a forecaster to the ensemble."""
        self.forecasters.append(forecaster)
        return self
    
    def fit(self, data: ForecastData, validation_split: float = 0.2, **kwargs) -> 'EnsembleForecaster':
        """
        Fit all forecasters in the ensemble.
        
        Parameters
        ----------
        data : ForecastData
            Training data
        validation_split : float
            Fraction of data to use for learning ensemble weights (if method='stacking')
        **kwargs
            Additional parameters passed to individual forecasters
        """
        if len(self.forecasters) == 0:
            raise ValueError("No forecasters added to ensemble")
        
        # Split data if using stacking
        if self.method == 'stacking' and validation_split > 0:
            train_data, val_data = data.split(1 - validation_split)
        else:
            train_data = data
            val_data = None
        
        # Fit all forecasters
        for forecaster in self.forecasters:
            forecaster.fit(train_data, **kwargs)
        
        # Learn ensemble weights if needed
        if self.method == 'stacking' and val_data is not None:
            self._fit_stacking(train_data, val_data)
        elif self.weights == 'learned':
            self._learn_weights(train_data)
        elif self.weights is None or self.weights == 'equal':
            self.weights = [1.0 / len(self.forecasters)] * len(self.forecasters)
        
        self.training_data = train_data
        self.is_fitted = True
        return self
    
    def _fit_stacking(self, train_data: ForecastData, val_data: ForecastData):
        """Fit stacking meta-model."""
        # Get predictions from all forecasters on validation set
        predictions = []
        for forecaster in self.forecasters:
            pred = forecaster.predict(horizon=len(val_data.values))
            predictions.append(pred.point_forecast)
        
        predictions = np.array(predictions).T  # Shape: (n_samples, n_forecasters)
        
        # Fit meta-model
        self.meta_model = LinearRegression()
        self.meta_model.fit(predictions, val_data.values)
    
    def _learn_weights(self, data: ForecastData):
        """Learn optimal weights using cross-validation."""
        # Simple approach: use inverse of validation error
        errors = []
        for forecaster in self.forecasters:
            # Simple validation: use last portion of data
            split_idx = int(len(data.values) * 0.8)
            train = ForecastData(
                data.timestamps[:split_idx],
                data.values[:split_idx]
            )
            val = ForecastData(
                data.timestamps[split_idx:],
                data.values[split_idx:]
            )
            
            forecaster.fit(train)
            pred = forecaster.predict(horizon=len(val.values))
            error = np.mean((pred.point_forecast - val.values) ** 2)
            errors.append(error)
        
        # Inverse error weighting (normalized)
        inv_errors = 1.0 / (np.array(errors) + 1e-10)
        self.weights = inv_errors / inv_errors.sum()
    
    def predict(
        self,
        horizon: Optional[int] = None,
        timestamps: Optional[Union[List, np.ndarray, pd.DatetimeIndex]] = None,
        return_quantiles: Optional[List[float]] = None,
        **kwargs
    ) -> ForecastResult:
        """
        Generate ensemble forecast.
        
        Parameters
        ----------
        horizon : int, optional
            Number of steps ahead
        timestamps : array-like, optional
            Timestamps to forecast
        return_quantiles : list of float, optional
            Quantile levels to return
        **kwargs
            Additional parameters
        
        Returns
        -------
        ForecastResult
            Ensemble forecast
        """
        if not self.is_fitted:
            raise ValueError("Forecaster must be fitted before prediction")
        
        # Get predictions from all forecasters
        all_predictions = []
        all_quantiles = {}
        
        for forecaster in self.forecasters:
            pred = forecaster.predict(
                horizon=horizon,
                timestamps=timestamps,
                return_quantiles=return_quantiles,
                **kwargs
            )
            all_predictions.append(pred.point_forecast)
            
            # Collect quantiles
            for q, values in pred.quantiles.items():
                if q not in all_quantiles:
                    all_quantiles[q] = []
                all_quantiles[q].append(values)
        
        all_predictions = np.array(all_predictions)  # Shape: (n_forecasters, n_samples)
        
        # Combine predictions based on method
        if self.method == 'weighted_average':
            weights = np.array(self.weights)
            ensemble_forecast = np.average(all_predictions, axis=0, weights=weights)
        elif self.method == 'stacking':
            if self.meta_model is None:
                warnings.warn("Meta-model not fitted, using weighted average")
                weights = np.array(self.weights)
                ensemble_forecast = np.average(all_predictions, axis=0, weights=weights)
            else:
                ensemble_forecast = self.meta_model.predict(all_predictions.T)
        elif self.method == 'median':
            ensemble_forecast = np.median(all_predictions, axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
        
        # Combine quantiles
        ensemble_quantiles = {}
        for q, quantile_predictions in all_quantiles.items():
            quantile_array = np.array(quantile_predictions)
            if self.method == 'weighted_average':
                weights = np.array(self.weights)
                ensemble_quantiles[q] = np.average(quantile_array, axis=0, weights=weights)
            elif self.method == 'median':
                ensemble_quantiles[q] = np.median(quantile_array, axis=0)
            else:
                ensemble_quantiles[q] = np.mean(quantile_array, axis=0)
        
        # Calculate uncertainty bounds
        lower_bound = None
        upper_bound = None
        if return_quantiles:
            if 0.05 in ensemble_quantiles:
                lower_bound = ensemble_quantiles[0.05]
            if 0.95 in ensemble_quantiles:
                upper_bound = ensemble_quantiles[0.95]
        
        return ForecastResult(
            point_forecast=ensemble_forecast,
            timestamps=timestamps,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            quantiles=ensemble_quantiles,
            metadata={'method': self.method, 'n_forecasters': len(self.forecasters)}
        )


class WeightedEnsemble(BaseForecaster):
    """
    Simplified weighted ensemble with automatic weight optimization.
    """
    
    def __init__(
        self,
        forecasters: List[BaseForecaster],
        optimization_method: str = 'inverse_variance',
        name: Optional[str] = None
    ):
        """
        Initialize weighted ensemble.
        
        Parameters
        ----------
        forecasters : list of BaseForecaster
            Forecasters to ensemble
        optimization_method : str
            Method for optimizing weights: 'inverse_variance', 'equal', or 'minimize_error'
        name : str, optional
            Name of the forecaster
        """
        super().__init__(name)
        self.forecasters = forecasters
        self.optimization_method = optimization_method
        self.weights = None
    
    def fit(self, data: ForecastData, **kwargs) -> 'WeightedEnsemble':
        """Fit all forecasters and optimize weights."""
        # Fit all forecasters
        for forecaster in self.forecasters:
            forecaster.fit(data, **kwargs)
        
        # Optimize weights
        if self.optimization_method == 'inverse_variance':
            self._optimize_inverse_variance(data)
        elif self.optimization_method == 'equal':
            self.weights = np.ones(len(self.forecasters)) / len(self.forecasters)
        elif self.optimization_method == 'minimize_error':
            self._optimize_minimize_error(data)
        else:
            self.weights = np.ones(len(self.forecasters)) / len(self.forecasters)
        
        self.training_data = data
        self.is_fitted = True
        return self
    
    def _optimize_inverse_variance(self, data: ForecastData):
        """Optimize weights using inverse variance method."""
        # Use cross-validation to estimate variance
        n_splits = min(5, len(data.values) // 10)
        if n_splits < 2:
            self.weights = np.ones(len(self.forecasters)) / len(self.forecasters)
            return
        
        variances = []
        for forecaster in self.forecasters:
            errors = []
            split_size = len(data.values) // n_splits
            
            for i in range(n_splits - 1):
                train_end = (i + 1) * split_size
                train = ForecastData(
                    data.timestamps[:train_end],
                    data.values[:train_end]
                )
                val = ForecastData(
                    data.timestamps[train_end:train_end + split_size],
                    data.values[train_end:train_end + split_size]
                )
                
                forecaster.fit(train)
                pred = forecaster.predict(horizon=len(val.values))
                errors.extend((pred.point_forecast - val.values) ** 2)
            
            variances.append(np.mean(errors))
        
        # Inverse variance weighting
        inv_var = 1.0 / (np.array(variances) + 1e-10)
        self.weights = inv_var / inv_var.sum()
    
    def _optimize_minimize_error(self, data: ForecastData):
        """Optimize weights by minimizing validation error."""
        from scipy.optimize import minimize
        
        # Use last portion for validation
        split_idx = int(len(data.values) * 0.8)
        train = ForecastData(
            data.timestamps[:split_idx],
            data.values[:split_idx]
        )
        val = ForecastData(
            data.timestamps[split_idx:],
            data.values[split_idx:]
        )
        
        # Get predictions
        predictions = []
        for forecaster in self.forecasters:
            forecaster.fit(train)
            pred = forecaster.predict(horizon=len(val.values))
            predictions.append(pred.point_forecast)
        
        predictions = np.array(predictions)
        
        # Objective function
        def objective(weights):
            weights = np.maximum(weights, 0)  # Ensure non-negative
            weights = weights / (weights.sum() + 1e-10)  # Normalize
            combined = np.dot(weights, predictions)
            return np.mean((combined - val.values) ** 2)
        
        # Optimize
        initial_weights = np.ones(len(self.forecasters)) / len(self.forecasters)
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=[(0, 1)] * len(self.forecasters),
                         constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1})
        
        self.weights = np.maximum(result.x, 0)
        self.weights = self.weights / (self.weights.sum() + 1e-10)
    
    def predict(
        self,
        horizon: Optional[int] = None,
        timestamps: Optional[Union[List, np.ndarray, pd.DatetimeIndex]] = None,
        return_quantiles: Optional[List[float]] = None,
        **kwargs
    ) -> ForecastResult:
        """Generate weighted ensemble forecast."""
        if not self.is_fitted:
            raise ValueError("Forecaster must be fitted before prediction")
        
        all_predictions = []
        for forecaster in self.forecasters:
            pred = forecaster.predict(
                horizon=horizon,
                timestamps=timestamps,
                return_quantiles=return_quantiles,
                **kwargs
            )
            all_predictions.append(pred.point_forecast)
        
        all_predictions = np.array(all_predictions)
        ensemble_forecast = np.average(all_predictions, axis=0, weights=self.weights)
        
        return ForecastResult(
            point_forecast=ensemble_forecast,
            timestamps=timestamps,
            metadata={'weights': self.weights.tolist()}
        )
