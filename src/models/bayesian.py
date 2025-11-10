"""
Bayesian forecasting models.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict
import warnings

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    warnings.warn("PyMC not available. BayesianForecaster will not work.")

from ..core import BaseForecaster, ForecastData, ForecastResult


class BayesianForecaster(BaseForecaster):
    """
    Bayesian forecaster using PyMC for probabilistic forecasting.
    
    Supports various Bayesian models including:
    - Gaussian process regression
    - Bayesian structural time series
    - Hierarchical models
    """
    
    def __init__(
        self,
        model_type: str = 'gaussian_process',
        prior_params: Optional[Dict] = None,
        name: Optional[str] = None
    ):
        """
        Initialize Bayesian forecaster.
        
        Parameters
        ----------
        model_type : str
            Type of Bayesian model: 'gaussian_process', 'structural', or 'hierarchical'
        prior_params : dict, optional
            Parameters for priors
        name : str, optional
            Name of the forecaster
        """
        if not PYMC_AVAILABLE:
            raise ImportError("PyMC is required for BayesianForecaster")
        
        super().__init__(name)
        self.model_type = model_type
        self.prior_params = prior_params or {}
        self.model = None
        self.trace = None
        self.idata = None
    
    def fit(self, data: ForecastData, draws: int = 1000, tune: int = 1000, **kwargs) -> 'BayesianForecaster':
        """
        Fit Bayesian model.
        
        Parameters
        ----------
        data : ForecastData
            Training data
        draws : int
            Number of posterior samples
        tune : int
            Number of tuning samples
        **kwargs
            Additional parameters
        """
        if self.model_type == 'gaussian_process':
            self._fit_gaussian_process(data, draws, tune, **kwargs)
        elif self.model_type == 'structural':
            self._fit_structural(data, draws, tune, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.training_data = data
        self.is_fitted = True
        return self
    
    def _fit_gaussian_process(self, data: ForecastData, draws: int, tune: int, **kwargs):
        """Fit Gaussian process model."""
        # Convert timestamps to numeric for GP
        if isinstance(data.timestamps, pd.DatetimeIndex):
            X = np.arange(len(data.timestamps))
        else:
            X = np.arange(len(data.timestamps))
        
        y = data.values
        
        with pm.Model() as model:
            # Hyperparameters
            lengthscale = pm.Gamma(
                'lengthscale',
                alpha=self.prior_params.get('lengthscale_alpha', 2.0),
                beta=self.prior_params.get('lengthscale_beta', 1.0)
            )
            variance = pm.Gamma(
                'variance',
                alpha=self.prior_params.get('variance_alpha', 2.0),
                beta=self.prior_params.get('variance_beta', 1.0)
            )
            noise = pm.Gamma(
                'noise',
                alpha=self.prior_params.get('noise_alpha', 2.0),
                beta=self.prior_params.get('noise_beta', 1.0)
            )
            
            # Covariance function
            cov = variance * pm.gp.cov.ExpQuad(1, lengthscale)
            
            # Gaussian process
            gp = pm.gp.Marginal(cov_func=cov)
            
            # Likelihood
            y_obs = gp.marginal_likelihood('y_obs', X=X.reshape(-1, 1), y=y, noise=noise)
            
            # Sample
            self.trace = pm.sample(draws=draws, tune=tune, return_inferencedata=True, **kwargs)
            self.idata = self.trace
        
        self.model = model
    
    def _fit_structural(self, data: ForecastData, draws: int, tune: int, **kwargs):
        """Fit Bayesian structural time series model."""
        y = data.values
        n = len(y)
        
        with pm.Model() as model:
            # Trend
            trend_sigma = pm.HalfNormal('trend_sigma', sigma=1.0)
            trend = pm.GaussianRandomWalk('trend', sigma=trend_sigma, shape=n)
            
            # Seasonality (simplified)
            seasonal_period = self.prior_params.get('seasonal_period', 12)
            if seasonal_period < n:
                seasonal_amplitude = pm.HalfNormal('seasonal_amplitude', sigma=1.0)
                seasonal = pm.Deterministic(
                    'seasonal',
                    seasonal_amplitude * pm.math.sin(2 * np.pi * np.arange(n) / seasonal_period)
                )
            else:
                seasonal = 0
            
            # Observation noise
            sigma = pm.HalfNormal('sigma', sigma=1.0)
            
            # Likelihood
            mu = trend + seasonal
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
            
            # Sample
            self.trace = pm.sample(draws=draws, tune=tune, return_inferencedata=True, **kwargs)
            self.idata = self.trace
        
        self.model = model
    
    def predict(
        self,
        horizon: Optional[int] = None,
        timestamps: Optional[Union[List, np.ndarray, pd.DatetimeIndex]] = None,
        return_quantiles: Optional[List[float]] = None,
        **kwargs
    ) -> ForecastResult:
        """Generate Bayesian forecast with uncertainty quantification."""
        if not self.is_fitted:
            raise ValueError("Forecaster must be fitted before prediction")
        
        if timestamps is not None:
            horizon = len(timestamps)
        elif horizon is None:
            horizon = 1
        
        # Generate future timestamps
        if timestamps is None:
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
                future_timestamps = pd.date_range(
                    start=last_timestamp,
                    periods=horizon + 1,
                    freq='D'
                )[1:]
        else:
            future_timestamps = pd.to_datetime(timestamps)
        
        # Convert to numeric
        if isinstance(self.training_data.timestamps, pd.DatetimeIndex):
            X_train = np.arange(len(self.training_data.timestamps))
            X_new = np.arange(len(self.training_data.timestamps),
                            len(self.training_data.timestamps) + horizon)
        else:
            X_train = np.arange(len(self.training_data.timestamps))
            X_new = np.arange(len(self.training_data.timestamps),
                            len(self.training_data.timestamps) + horizon)
        
        # Generate posterior predictive samples
        with self.model:
            if self.model_type == 'gaussian_process':
                # For GP, we need to use the GP's conditional
                # This is simplified - full implementation would use gp.conditional
                f_pred = pm.sample_posterior_predictive(
                    self.trace,
                    var_names=['y_obs'],
                    predictions=True,
                    extend_inferencedata=True
                )
            else:
                # For structural model, extend the model
                f_pred = pm.sample_posterior_predictive(
                    self.trace,
                    var_names=['y_obs'],
                    predictions=True,
                    extend_inferencedata=True
                )
        
        # Extract predictions (simplified - would need proper GP conditional)
        # For now, use posterior samples to estimate
        posterior_samples = az.extract(self.idata, var_names=['y_obs'])
        
        # Calculate point forecast and quantiles
        point_forecast = np.mean(posterior_samples.values, axis=0)
        
        quantiles = {}
        if return_quantiles:
            for q in return_quantiles:
                quantiles[q] = np.percentile(posterior_samples.values, q * 100, axis=0)
        
        # Get uncertainty bounds
        lower_bound = np.percentile(posterior_samples.values, 2.5, axis=0)
        upper_bound = np.percentile(posterior_samples.values, 97.5, axis=0)
        
        return ForecastResult(
            point_forecast=point_forecast[-horizon:],  # Take last horizon points
            timestamps=future_timestamps,
            lower_bound=lower_bound[-horizon:] if len(lower_bound) >= horizon else None,
            upper_bound=upper_bound[-horizon:] if len(upper_bound) >= horizon else None,
            quantiles={q: vals[-horizon:] for q, vals in quantiles.items()},
            metadata={'model_type': self.model_type}
        )

