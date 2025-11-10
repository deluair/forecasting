"""
Econometric forecasting models for advanced economic analysis.

Includes Vector Autoregression (VAR), Vector Error Correction Models (VECM),
and State Space models suitable for macroeconomic forecasting.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Tuple
import warnings

try:
    from statsmodels.tsa.vector_ar.var_model import VAR
    from statsmodels.tsa.vector_ar.vecm import VECM
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Econometric models will not work.")

from ..core import BaseForecaster, ForecastData, ForecastResult


class VARForecaster(BaseForecaster):
    """
    Vector Autoregression (VAR) forecaster for multivariate time series.
    
    Suitable for modeling relationships between multiple economic variables
    (e.g., GDP, inflation, interest rates, exchange rates).
    
    References:
    - Lütkepohl, H. (2005). New Introduction to Multiple Time Series Analysis.
    - Sims, C. A. (1980). Macroeconomics and reality.
    """
    
    def __init__(
        self,
        maxlags: int = 4,
        ic: str = 'aic',
        trend: str = 'c',
        name: Optional[str] = None
    ):
        """
        Initialize VAR forecaster.
        
        Parameters
        ----------
        maxlags : int
            Maximum number of lags to consider
        ic : str
            Information criterion for lag selection: 'aic', 'bic', 'fpe', 'hqic'
        trend : str
            Trend specification: 'c' (constant), 'ct' (constant+trend), 'ctt' (constant+linear+quadratic)
        name : str, optional
            Name of the forecaster
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for VARForecaster")
        
        super().__init__(name)
        self.maxlags = maxlags
        self.ic = ic
        self.trend = trend
        self.model = None
        self.selected_lags = None
        self.variable_names = None
    
    def fit(self, data: ForecastData, **kwargs) -> 'VARForecaster':
        """
        Fit VAR model to multivariate time series data.
        
        Parameters
        ----------
        data : ForecastData
            Training data. Values should be 2D array (n_obs x n_vars)
            or DataFrame-like structure
        
        Returns
        -------
        self : VARForecaster
        """
        # Convert to DataFrame if needed
        if isinstance(data.values, np.ndarray):
            if data.values.ndim == 1:
                raise ValueError("VAR requires multivariate data (2D array)")
            df = pd.DataFrame(data.values, index=data.timestamps)
        else:
            df = pd.DataFrame(data.values)
        
        self.variable_names = df.columns.tolist() if hasattr(df, 'columns') else None
        
        # Fit VAR model
        self.model = VAR(df)
        self.selected_lags = self.model.select_order(maxlags=self.maxlags, verbose=False)
        lag_order = getattr(self.selected_lags, self.ic)
        
        self.model = self.model.fit(maxlags=lag_order, ic=self.ic, trend=self.trend)
        
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
        """
        Generate VAR forecast.
        
        Parameters
        ----------
        horizon : int, optional
            Number of steps ahead to forecast
        timestamps : array-like, optional
            Specific timestamps to forecast
        return_quantiles : list of float, optional
            Quantile levels to return
        
        Returns
        -------
        ForecastResult
            Forecast results
        """
        if not self.is_fitted:
            raise ValueError("Forecaster must be fitted before prediction")
        
        if horizon is None and timestamps is None:
            horizon = 1
        
        if timestamps is not None:
            horizon = len(timestamps)
        
        # Generate forecast
        forecast_result = self.model.forecast(
            self.model.y,
            steps=horizon,
            alpha=kwargs.get('alpha', 0.05)
        )
        
        # Extract point forecasts (first variable if multivariate)
        if forecast_result.ndim == 2:
            point_forecast = forecast_result[:, 0]  # First variable
        else:
            point_forecast = forecast_result
        
        # Get forecast intervals
        forecast_ci = self.model.forecast_interval(
            self.model.y,
            steps=horizon,
            alpha=kwargs.get('alpha', 0.05)
        )
        
        lower_bound = forecast_ci[:, 0, 0] if forecast_ci.ndim == 3 else None
        upper_bound = forecast_ci[:, 1, 0] if forecast_ci.ndim == 3 else None
        
        # Generate timestamps
        if timestamps is None:
            last_timestamp = self.training_data.timestamps.iloc[-1] if hasattr(self.training_data.timestamps, 'iloc') else self.training_data.timestamps[-1]
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
        
        return ForecastResult(
            point_forecast=point_forecast,
            timestamps=timestamps,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            metadata={
                'model_type': 'VAR',
                'lags': self.selected_lags.aic if self.selected_lags else None,
                'n_variables': forecast_result.shape[1] if forecast_result.ndim == 2 else 1
            }
        )
    
    def impulse_response(self, periods: int = 10, orth: bool = True) -> np.ndarray:
        """
        Compute impulse response functions.
        
        Parameters
        ----------
        periods : int
            Number of periods ahead
        orth : bool
            If True, use orthogonalized IRF (Cholesky decomposition)
        
        Returns
        -------
        irf : np.ndarray
            Impulse response functions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        irf = self.model.irf(periods=periods, orth=orth)
        return irf.irfs
    
    def forecast_error_variance_decomposition(self, periods: int = 10) -> np.ndarray:
        """
        Compute forecast error variance decomposition (FEVD).
        
        Parameters
        ----------
        periods : int
            Number of periods ahead
        
        Returns
        -------
        fevd : np.ndarray
            Forecast error variance decomposition
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        irf = self.model.irf(periods=periods)
        fevd = irf.fevd()
        return fevd.decomp


class VECMForecaster(BaseForecaster):
    """
    Vector Error Correction Model (VECM) forecaster.
    
    Suitable for cointegrated time series (e.g., long-run economic relationships).
    
    References:
    - Engle, R. F., & Granger, C. W. J. (1987). Co-integration and error correction.
    - Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors.
    """
    
    def __init__(
        self,
        k_ar_diff: int = 1,
        coint_rank: Optional[int] = None,
        deterministic: str = 'ci',
        name: Optional[str] = None
    ):
        """
        Initialize VECM forecaster.
        
        Parameters
        ----------
        k_ar_diff : int
            Number of lagged differences
        coint_rank : int, optional
            Cointegration rank. If None, will be determined via Johansen test
        deterministic : str
            Deterministic terms: 'ci' (constant in cointegrating), 'co' (constant outside), 'lo' (linear trend)
        name : str, optional
            Name of the forecaster
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for VECMForecaster")
        
        super().__init__(name)
        self.k_ar_diff = k_ar_diff
        self.coint_rank = coint_rank
        self.deterministic = deterministic
        self.model = None
    
    def fit(self, data: ForecastData, **kwargs) -> 'VECMForecaster':
        """Fit VECM model."""
        # Convert to DataFrame
        if isinstance(data.values, np.ndarray):
            if data.values.ndim == 1:
                raise ValueError("VECM requires multivariate data")
            df = pd.DataFrame(data.values, index=data.timestamps)
        else:
            df = pd.DataFrame(data.values)
        
        # Fit VECM
        self.model = VECM(
            df,
            k_ar_diff=self.k_ar_diff,
            coint_rank=self.coint_rank,
            deterministic=self.deterministic
        )
        self.model = self.model.fit()
        
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
        """Generate VECM forecast."""
        if not self.is_fitted:
            raise ValueError("Forecaster must be fitted before prediction")
        
        if horizon is None and timestamps is None:
            horizon = 1
        
        if timestamps is not None:
            horizon = len(timestamps)
        
        # Generate forecast
        forecast = self.model.predict(steps=horizon)
        
        # Extract first variable
        point_forecast = forecast[:, 0] if forecast.ndim == 2 else forecast
        
        # Generate timestamps
        if timestamps is None:
            last_timestamp = self.training_data.timestamps.iloc[-1] if hasattr(self.training_data.timestamps, 'iloc') else self.training_data.timestamps[-1]
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
        
        return ForecastResult(
            point_forecast=point_forecast,
            timestamps=timestamps,
            metadata={
                'model_type': 'VECM',
                'coint_rank': self.model.coint_rank,
                'k_ar_diff': self.k_ar_diff
            }
        )


class StateSpaceForecaster(BaseForecaster):
    """
    State Space Model forecaster using Kalman filtering.
    
    Suitable for structural time series models and unobserved components.
    
    References:
    - Durbin, J., & Koopman, S. J. (2012). Time Series Analysis by State Space Methods.
    - Harvey, A. C. (1990). Forecasting, Structural Time Series Models and the Kalman Filter.
    """
    
    def __init__(
        self,
        trend: bool = True,
        seasonal: Optional[int] = None,
        cycle: bool = False,
        name: Optional[str] = None
    ):
        """
        Initialize State Space forecaster.
        
        Parameters
        ----------
        trend : bool
            Include trend component
        seasonal : int, optional
            Seasonal period (e.g., 12 for monthly, 4 for quarterly)
        cycle : bool
            Include cycle component
        name : str, optional
            Name of the forecaster
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for StateSpaceForecaster")
        
        super().__init__(name)
        self.trend = trend
        self.seasonal = seasonal
        self.cycle = cycle
        self.model = None
    
    def fit(self, data: ForecastData, **kwargs) -> 'StateSpaceForecaster':
        """Fit State Space model."""
        # For now, use SARIMAX as proxy for state space
        # Full implementation would use UnobservedComponents
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, self.seasonal) if self.seasonal else None
        
        self.model = SARIMAX(
            data.values,
            order=order,
            seasonal_order=seasonal_order,
            trend='c' if self.trend else 'n'
        )
        self.model = self.model.fit(disp=False)
        
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
        """Generate State Space forecast."""
        if not self.is_fitted:
            raise ValueError("Forecaster must be fitted before prediction")
        
        if horizon is None and timestamps is None:
            horizon = 1
        
        if timestamps is not None:
            horizon = len(timestamps)
        
        # Generate forecast
        forecast_result = self.model.get_forecast(steps=horizon)
        point_forecast = forecast_result.predicted_mean.values if hasattr(forecast_result.predicted_mean, 'values') else np.asarray(forecast_result.predicted_mean)
        
        # Get confidence intervals
        conf_int = forecast_result.conf_int()
        if isinstance(conf_int, pd.DataFrame):
            lower_bound = conf_int.iloc[:, 0].values
            upper_bound = conf_int.iloc[:, 1].values
        else:
            lower_bound = conf_int[:, 0]
            upper_bound = conf_int[:, 1]
        
        # Generate timestamps
        if timestamps is None:
            last_timestamp = self.training_data.timestamps.iloc[-1] if hasattr(self.training_data.timestamps, 'iloc') else self.training_data.timestamps[-1]
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
        
        return ForecastResult(
            point_forecast=point_forecast,
            timestamps=timestamps,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            metadata={
                'model_type': 'StateSpace',
                'trend': self.trend,
                'seasonal': self.seasonal
            }
        )

