"""
Utility functions for forecasting.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict, Tuple
from scipy import stats
import warnings

from ..core import ForecastData, ForecastResult


class CalibrationTool:
    """Tools for calibrating probabilistic forecasts."""
    
    @staticmethod
    def isotonic_calibration(
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> Tuple[np.ndarray, callable]:
        """
        Apply isotonic regression for calibration.
        
        Parameters
        ----------
        predictions : np.ndarray
            Uncalibrated predictions
        actuals : np.ndarray
            Actual binary outcomes
        
        Returns
        -------
        calibrated_predictions : np.ndarray
            Calibrated predictions
        calibration_function : callable
            Function to apply calibration to new predictions
        """
        from sklearn.isotonic import IsotonicRegression
        
        # Fit isotonic regression
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(predictions, actuals)
        
        # Calibrate predictions
        calibrated = iso_reg.transform(predictions)
        
        return calibrated, iso_reg.transform
    
    @staticmethod
    def platt_scaling(
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> Tuple[np.ndarray, callable]:
        """
        Apply Platt scaling (logistic regression) for calibration.
        
        Parameters
        ----------
        predictions : np.ndarray
            Uncalibrated predictions
        actuals : np.ndarray
            Actual binary outcomes
        
        Returns
        -------
        calibrated_predictions : np.ndarray
            Calibrated predictions
        calibration_function : callable
            Function to apply calibration to new predictions
        """
        from sklearn.linear_model import LogisticRegression
        
        # Fit logistic regression
        lr = LogisticRegression()
        lr.fit(predictions.reshape(-1, 1), actuals)
        
        # Calibrate predictions
        calibrated = lr.predict_proba(predictions.reshape(-1, 1))[:, 1]
        
        def calibrate_func(x):
            return lr.predict_proba(np.array(x).reshape(-1, 1))[:, 1]
        
        return calibrated, calibrate_func


class UncertaintyQuantifier:
    """Tools for quantifying forecast uncertainty."""
    
    @staticmethod
    def bootstrap_uncertainty(
        forecaster,
        data: ForecastData,
        n_samples: int = 100,
        horizon: int = 1,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate uncertainty using bootstrap sampling.
        
        Parameters
        ----------
        forecaster : BaseForecaster
            Fitted forecaster
        data : ForecastData
            Training data
        n_samples : int
            Number of bootstrap samples
        horizon : int
            Forecast horizon
        confidence : float
            Confidence level
        
        Returns
        -------
        point_forecast : np.ndarray
            Point forecast (mean)
        lower_bound : np.ndarray
            Lower confidence bound
        upper_bound : np.ndarray
            Upper confidence bound
        """
        predictions = []
        
        for _ in range(n_samples):
            # Bootstrap sample
            n = len(data.values)
            indices = np.random.choice(n, size=n, replace=True)
            boot_data = ForecastData(
                data.timestamps[indices],
                data.values[indices]
            )
            
            # Fit and predict
            forecaster.fit(boot_data)
            pred = forecaster.predict(horizon=horizon)
            predictions.append(pred.point_forecast)
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        point_forecast = np.mean(predictions, axis=0)
        alpha = 1 - confidence
        lower_bound = np.percentile(predictions, alpha / 2 * 100, axis=0)
        upper_bound = np.percentile(predictions, (1 - alpha / 2) * 100, axis=0)
        
        return point_forecast, lower_bound, upper_bound
    
    @staticmethod
    def conformal_prediction(
        forecaster,
        train_data: ForecastData,
        calibration_data: ForecastData,
        horizon: int = 1,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate uncertainty using conformal prediction.
        
        Parameters
        ----------
        forecaster : BaseForecaster
            Forecaster
        train_data : ForecastData
            Training data
        calibration_data : ForecastData
            Calibration data
        horizon : int
            Forecast horizon
        confidence : float
            Confidence level
        
        Returns
        -------
        point_forecast : np.ndarray
            Point forecast
        lower_bound : np.ndarray
            Lower conformal bound
        upper_bound : np.ndarray
            Upper conformal bound
        """
        # Fit on training data
        forecaster.fit(train_data)
        
        # Get predictions on calibration data
        pred_cal = forecaster.predict(horizon=len(calibration_data.values))
        
        # Calculate residuals
        residuals = np.abs(calibration_data.values - pred_cal.point_forecast)
        
        # Get quantile of residuals
        alpha = 1 - confidence
        quantile = np.quantile(residuals, 1 - alpha)
        
        # Get forecast
        pred = forecaster.predict(horizon=horizon)
        point_forecast = pred.point_forecast
        
        # Calculate bounds
        lower_bound = point_forecast - quantile
        upper_bound = point_forecast + quantile
        
        return point_forecast, lower_bound, upper_bound


class FeatureEngineering:
    """Feature engineering utilities for forecasting."""
    
    @staticmethod
    def create_time_features(
        timestamps: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Create time-based features from timestamps.
        
        Parameters
        ----------
        timestamps : pd.DatetimeIndex
            Timestamps
        
        Returns
        -------
        features : pd.DataFrame
            Time features
        """
        features = pd.DataFrame(index=timestamps)
        
        features['year'] = timestamps.year
        features['month'] = timestamps.month
        features['day'] = timestamps.day
        features['dayofweek'] = timestamps.dayofweek
        features['dayofyear'] = timestamps.dayofyear
        features['week'] = timestamps.isocalendar().week
        features['quarter'] = timestamps.quarter
        
        # Cyclical encoding
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        features['dayofweek_sin'] = np.sin(2 * np.pi * features['dayofweek'] / 7)
        features['dayofweek_cos'] = np.cos(2 * np.pi * features['dayofweek'] / 7)
        
        return features
    
    @staticmethod
    def create_lag_features(
        values: np.ndarray,
        lags: List[int]
    ) -> pd.DataFrame:
        """
        Create lagged features.
        
        Parameters
        ----------
        values : np.ndarray
            Time series values
        lags : list of int
            Lag values to create
        
        Returns
        -------
        features : pd.DataFrame
            Lag features
        """
        features = {}
        for lag in lags:
            lagged = np.roll(values, lag)
            lagged[:lag] = np.nan
            features[f'lag_{lag}'] = lagged
        
        return pd.DataFrame(features)
    
    @staticmethod
    def create_rolling_features(
        values: np.ndarray,
        windows: List[int],
        functions: List[str] = ['mean', 'std']
    ) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Parameters
        ----------
        values : np.ndarray
            Time series values
        windows : list of int
            Window sizes
        functions : list of str
            Functions to apply: 'mean', 'std', 'min', 'max'
        
        Returns
        -------
        features : pd.DataFrame
            Rolling features
        """
        df = pd.Series(values)
        features = {}
        
        for window in windows:
            for func in functions:
                if func == 'mean':
                    features[f'rolling_{window}_{func}'] = df.rolling(window).mean()
                elif func == 'std':
                    features[f'rolling_{window}_{func}'] = df.rolling(window).std()
                elif func == 'min':
                    features[f'rolling_{window}_{func}'] = df.rolling(window).min()
                elif func == 'max':
                    features[f'rolling_{window}_{func}'] = df.rolling(window).max()
        
        return pd.DataFrame(features)
