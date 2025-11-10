"""
Time series forecasting models.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List
import warnings

try:
    from statsmodels.tsa.arima.model import ARIMA
    from pmdarima import auto_arima
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    warnings.warn("statsmodels or pmdarima not available. ARIMA models will not work.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not available. ProphetForecaster will not work.")

from ..core import BaseForecaster, ForecastData, ForecastResult


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA (AutoRegressive Integrated Moving Average) forecaster.
    """
    
    def __init__(
        self,
        order: Optional[tuple] = None,
        seasonal_order: Optional[tuple] = None,
        auto: bool = True,
        name: Optional[str] = None
    ):
        """
        Initialize ARIMA forecaster.
        
        Parameters
        ----------
        order : tuple, optional
            (p, d, q) order for ARIMA model
        seasonal_order : tuple, optional
            (P, D, Q, s) seasonal order
        auto : bool
            If True, automatically select best order using auto_arima
        name : str, optional
            Name of the forecaster
        """
        if not ARIMA_AVAILABLE:
            raise ImportError("statsmodels and pmdarima are required for ARIMAForecaster")
        
        super().__init__(name)
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto = auto
        self.model = None
        self.auto_model = None
    
    def fit(self, data: ForecastData, **kwargs) -> 'ARIMAForecaster':
        """Fit ARIMA model."""
        if self.auto:
            # Use auto_arima to find best parameters
            self.auto_model = auto_arima(
                data.values,
                start_p=kwargs.get('start_p', 0),
                start_d=kwargs.get('start_d', 0),
                start_q=kwargs.get('start_q', 0),
                max_p=kwargs.get('max_p', 5),
                max_d=kwargs.get('max_d', 2),
                max_q=kwargs.get('max_q', 5),
                seasonal=kwargs.get('seasonal', False),
                stepwise=kwargs.get('stepwise', True),
                suppress_warnings=True,
                error_action='ignore'
            )
            self.order = self.auto_model.order
            self.seasonal_order = self.auto_model.seasonal_order
            self.model = self.auto_model.arima_res_
        else:
            # Use specified order
            self.model = ARIMA(
                data.values,
                order=self.order or (1, 1, 1),
                seasonal_order=self.seasonal_order
            ).fit()
        
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
        """Generate ARIMA forecast."""
        if not self.is_fitted:
            raise ValueError("Forecaster must be fitted before prediction")
        
        if horizon is None and timestamps is None:
            horizon = 1
        
        if timestamps is not None:
            horizon = len(timestamps)
        
        # Get forecast
        forecast_result = self.model.get_forecast(steps=horizon)
        # Handle both Series and array returns
        if hasattr(forecast_result.predicted_mean, 'values'):
            point_forecast = forecast_result.predicted_mean.values
        else:
            point_forecast = np.asarray(forecast_result.predicted_mean)
        
        # Get confidence intervals
        conf_int = forecast_result.conf_int()
        if isinstance(conf_int, pd.DataFrame):
            lower_bound = conf_int.iloc[:, 0].values
            upper_bound = conf_int.iloc[:, 1].values
        else:
            # Handle numpy array case
            conf_int = np.asarray(conf_int)
            lower_bound = conf_int[:, 0]
            upper_bound = conf_int[:, 1]
        
        # Generate timestamps if not provided
        if timestamps is None:
            last_timestamp = self.training_data.timestamps.iloc[-1] if hasattr(self.training_data.timestamps, 'iloc') else self.training_data.timestamps[-1]
            if isinstance(self.training_data.timestamps, pd.DatetimeIndex):
                freq = pd.infer_freq(self.training_data.timestamps)
                if freq is None:
                    # Default to daily
                    timestamps = pd.date_range(
                        start=last_timestamp,
                        periods=horizon + 1,
                        freq='D'
                    )[1:]
                else:
                    timestamps = pd.date_range(
                        start=last_timestamp,
                        periods=horizon + 1,
                        freq=freq
                    )[1:]
            else:
                timestamps = None
        
        # Calculate quantiles if requested
        quantiles = {}
        if return_quantiles:
            # Use normal approximation for quantiles
            std = (upper_bound - lower_bound) / (2 * 1.96)  # Assuming 95% CI
            for q in return_quantiles:
                from scipy import stats
                z_score = stats.norm.ppf(q)
                quantiles[q] = point_forecast + z_score * std
        
        return ForecastResult(
            point_forecast=point_forecast,
            timestamps=timestamps,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            quantiles=quantiles,
            metadata={'order': self.order, 'seasonal_order': self.seasonal_order}
        )


class ProphetForecaster(BaseForecaster):
    """
    Facebook Prophet forecaster for time series with seasonality.
    """
    
    def __init__(
        self,
        growth: str = 'linear',
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        seasonality_mode: str = 'additive',
        name: Optional[str] = None,
        **prophet_kwargs
    ):
        """
        Initialize Prophet forecaster.
        
        Parameters
        ----------
        growth : str
            'linear' or 'logistic' growth trend
        yearly_seasonality : bool
            Fit yearly seasonality
        weekly_seasonality : bool
            Fit weekly seasonality
        daily_seasonality : bool
            Fit daily seasonality
        seasonality_mode : str
            'additive' or 'multiplicative'
        name : str, optional
            Name of the forecaster
        **prophet_kwargs
            Additional parameters passed to Prophet
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is required for ProphetForecaster")
        
        super().__init__(name)
        self.growth = growth
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.prophet_kwargs = prophet_kwargs
        self.model = None
    
    def fit(self, data: ForecastData, **kwargs) -> 'ProphetForecaster':
        """Fit Prophet model."""
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        df = pd.DataFrame({
            'ds': data.timestamps,
            'y': data.values
        })
        
        # Initialize and fit Prophet
        self.model = Prophet(
            growth=self.growth,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            **self.prophet_kwargs
        )
        
        self.model.fit(df)
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
        """Generate Prophet forecast."""
        if not self.is_fitted:
            raise ValueError("Forecaster must be fitted before prediction")
        
        if timestamps is not None:
            future_df = pd.DataFrame({'ds': pd.to_datetime(timestamps)})
        elif horizon is not None:
            # Generate future timestamps
            last_timestamp = self.training_data.timestamps[-1]
            if isinstance(self.training_data.timestamps, pd.DatetimeIndex):
                freq = pd.infer_freq(self.training_data.timestamps)
                if freq is None:
                    freq = 'D'
                future_timestamps = pd.date_range(
                    start=last_timestamp,
                    periods=horizon + 1,
                    freq=freq
                )[1:]
            else:
                # Fallback to daily
                future_timestamps = pd.date_range(
                    start=last_timestamp,
                    periods=horizon + 1,
                    freq='D'
                )[1:]
            future_df = pd.DataFrame({'ds': future_timestamps})
        else:
            horizon = 1
            last_timestamp = self.training_data.timestamps[-1]
            future_timestamps = pd.date_range(
                start=last_timestamp,
                periods=2,
                freq='D'
            )[1:]
            future_df = pd.DataFrame({'ds': future_timestamps})
        
        # Make forecast
        forecast_df = self.model.predict(future_df)
        
        point_forecast = forecast_df['yhat'].values
        lower_bound = forecast_df['yhat_lower'].values
        upper_bound = forecast_df['yhat_upper'].values
        
        # Extract quantiles if requested
        quantiles = {}
        if return_quantiles:
            # Prophet provides uncertainty intervals, approximate quantiles
            std = (upper_bound - lower_bound) / (2 * 1.96)
            from scipy import stats
            for q in return_quantiles:
                z_score = stats.norm.ppf(q)
                quantiles[q] = point_forecast + z_score * std
        
        timestamps = future_df['ds'].values
        
        return ForecastResult(
            point_forecast=point_forecast,
            timestamps=timestamps,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            quantiles=quantiles,
            metadata={'growth': self.growth, 'seasonality_mode': self.seasonality_mode}
        )

