"""
Evaluation metrics for forecasting.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict
from scipy import stats
import warnings

from ..core import ForecastResult, ForecastData


class BaseMetric:
    """Base class for evaluation metrics."""
    
    def __init__(self, name: Optional[str] = None):
        """Initialize metric."""
        self.name = name or self.__class__.__name__
    
    def evaluate(
        self,
        predictions: Union[ForecastResult, np.ndarray],
        actuals: Union[ForecastData, np.ndarray],
        **kwargs
    ) -> float:
        """
        Evaluate predictions against actuals.
        
        Parameters
        ----------
        predictions : ForecastResult or np.ndarray
            Predictions
        actuals : ForecastData or np.ndarray
            Actual values
        
        Returns
        -------
        score : float
            Metric score
        """
        raise NotImplementedError("Subclasses must implement evaluate()")
    
    def __call__(self, predictions, actuals, **kwargs):
        """Make metric callable."""
        return self.evaluate(predictions, actuals, **kwargs)


class BrierScore(BaseMetric):
    """
    Brier Score for probabilistic forecasts.
    
    For binary outcomes: BS = mean((predicted_prob - actual)^2)
    Lower is better.
    """
    
    def evaluate(
        self,
        predictions: Union[ForecastResult, np.ndarray],
        actuals: Union[ForecastData, np.ndarray],
        **kwargs
    ) -> float:
        """Calculate Brier Score."""
        # Extract values
        if isinstance(predictions, ForecastResult):
            pred_values = predictions.point_forecast
        else:
            pred_values = np.asarray(predictions)
        
        if isinstance(actuals, ForecastData):
            actual_values = actuals.values
        else:
            actual_values = np.asarray(actuals)
        
        # Ensure binary (0/1)
        actual_values = np.clip(actual_values, 0, 1)
        pred_values = np.clip(pred_values, 0, 1)
        
        # Calculate Brier Score
        brier = np.mean((pred_values - actual_values) ** 2)
        return float(brier)


class LogScore(BaseMetric):
    """
    Logarithmic Score (Log Score) for probabilistic forecasts.
    
    LS = -mean(log(predicted_probability_of_actual_outcome)))
    Higher is better (less negative).
    """
    
    def evaluate(
        self,
        predictions: Union[ForecastResult, np.ndarray],
        actuals: Union[ForecastData, np.ndarray],
        **kwargs
    ) -> float:
        """Calculate Log Score."""
        # Extract values
        if isinstance(predictions, ForecastResult):
            pred_values = predictions.point_forecast
        else:
            pred_values = np.asarray(predictions)
        
        if isinstance(actuals, ForecastData):
            actual_values = actuals.values
        else:
            actual_values = np.asarray(actuals)
        
        # Ensure probabilities are in valid range
        epsilon = 1e-10
        pred_values = np.clip(pred_values, epsilon, 1 - epsilon)
        
        # For binary outcomes
        if np.all(np.isin(actual_values, [0, 1])):
            # Log score: -log(p) if actual=1, -log(1-p) if actual=0
            log_scores = np.where(
                actual_values == 1,
                -np.log(pred_values),
                -np.log(1 - pred_values)
            )
        else:
            # For continuous outcomes, use probability density
            # This is simplified - full implementation would use PDF
            warnings.warn("Log score for continuous outcomes not fully implemented")
            log_scores = -np.log(pred_values + epsilon)
        
        return float(np.mean(log_scores))


class MAE(BaseMetric):
    """Mean Absolute Error."""
    
    def evaluate(
        self,
        predictions: Union[ForecastResult, np.ndarray],
        actuals: Union[ForecastData, np.ndarray],
        **kwargs
    ) -> float:
        """Calculate MAE."""
        if isinstance(predictions, ForecastResult):
            pred_values = predictions.point_forecast
        else:
            pred_values = np.asarray(predictions)
        
        if isinstance(actuals, ForecastData):
            actual_values = actuals.values
        else:
            actual_values = np.asarray(actuals)
        
        return float(np.mean(np.abs(pred_values - actual_values)))


class RMSE(BaseMetric):
    """Root Mean Squared Error."""
    
    def evaluate(
        self,
        predictions: Union[ForecastResult, np.ndarray],
        actuals: Union[ForecastData, np.ndarray],
        **kwargs
    ) -> float:
        """Calculate RMSE."""
        if isinstance(predictions, ForecastResult):
            pred_values = predictions.point_forecast
        else:
            pred_values = np.asarray(predictions)
        
        if isinstance(actuals, ForecastData):
            actual_values = actuals.values
        else:
            actual_values = np.asarray(actuals)
        
        return float(np.sqrt(np.mean((pred_values - actual_values) ** 2)))


class MAPE(BaseMetric):
    """Mean Absolute Percentage Error."""
    
    def evaluate(
        self,
        predictions: Union[ForecastResult, np.ndarray],
        actuals: Union[ForecastData, np.ndarray],
        **kwargs
    ) -> float:
        """Calculate MAPE."""
        if isinstance(predictions, ForecastResult):
            pred_values = predictions.point_forecast
        else:
            pred_values = np.asarray(predictions)
        
        if isinstance(actuals, ForecastData):
            actual_values = actuals.values
        else:
            actual_values = np.asarray(actuals)
        
        # Avoid division by zero
        epsilon = 1e-10
        mape = np.mean(np.abs((actual_values - pred_values) / (actual_values + epsilon))) * 100
        return float(mape)


class CalibrationScore(BaseMetric):
    """
    Calibration score - measures how well-calibrated probabilistic forecasts are.
    
    Uses reliability diagram approach.
    """
    
    def __init__(self, n_bins: int = 10, name: Optional[str] = None):
        """
        Initialize calibration score.
        
        Parameters
        ----------
        n_bins : int
            Number of bins for calibration curve
        """
        super().__init__(name)
        self.n_bins = n_bins
    
    def evaluate(
        self,
        predictions: Union[ForecastResult, np.ndarray],
        actuals: Union[ForecastData, np.ndarray],
        **kwargs
    ) -> float:
        """
        Calculate calibration score (ECE - Expected Calibration Error).
        
        Lower is better.
        """
        if isinstance(predictions, ForecastResult):
            pred_values = predictions.point_forecast
        else:
            pred_values = np.asarray(predictions)
        
        if isinstance(actuals, ForecastData):
            actual_values = actuals.values
        else:
            actual_values = np.asarray(actuals)
        
        # Ensure binary
        actual_values = np.clip(actual_values, 0, 1)
        pred_values = np.clip(pred_values, 0, 1)
        
        # Bin predictions
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (pred_values > bin_lower) & (pred_values <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = actual_values[in_bin].mean()
                # Average confidence in this bin
                avg_confidence_in_bin = pred_values[in_bin].mean()
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)


class SharpnessScore(BaseMetric):
    """
    Sharpness score - measures how concentrated predictions are.
    
    For probabilistic forecasts, lower variance = sharper (better).
    """
    
    def evaluate(
        self,
        predictions: Union[ForecastResult, np.ndarray],
        actuals: Optional[Union[ForecastData, np.ndarray]] = None,
        **kwargs
    ) -> float:
        """
        Calculate sharpness (variance of predictions).
        
        Lower is better (sharper predictions).
        """
        if isinstance(predictions, ForecastResult):
            pred_values = predictions.point_forecast
        else:
            pred_values = np.asarray(predictions)
        
        # Calculate variance
        sharpness = np.var(pred_values)
        return float(sharpness)


class CRPS(BaseMetric):
    """
    Continuous Ranked Probability Score.
    
    Measures the quality of probabilistic forecasts.
    Lower is better.
    """
    
    def evaluate(
        self,
        predictions: Union[ForecastResult, np.ndarray],
        actuals: Union[ForecastData, np.ndarray],
        **kwargs
    ) -> float:
        """
        Calculate CRPS.
        
        For point forecasts, CRPS reduces to MAE.
        For probabilistic forecasts, integrates over all thresholds.
        """
        if isinstance(predictions, ForecastResult):
            pred_values = predictions.point_forecast
            # If quantiles available, use them
            if predictions.quantiles:
                # Use quantiles for better CRPS calculation
                quantiles = sorted(predictions.quantiles.keys())
                quantile_values = [predictions.quantiles[q] for q in quantiles]
                # Approximate CRPS using quantiles
                # This is simplified - full implementation would integrate
                pass
        else:
            pred_values = np.asarray(predictions)
        
        if isinstance(actuals, ForecastData):
            actual_values = actuals.values
        else:
            actual_values = np.asarray(actuals)
        
        # Simplified CRPS: for point forecasts, it's MAE
        # Full implementation would use CDF
        crps = np.mean(np.abs(pred_values - actual_values))
        return float(crps)


class MetricSuite:
    """Collection of metrics for comprehensive evaluation."""
    
    def __init__(self, metrics: Optional[List[BaseMetric]] = None):
        """
        Initialize metric suite.
        
        Parameters
        ----------
        metrics : list of BaseMetric, optional
            List of metrics to compute. If None, uses default set.
        """
        if metrics is None:
            self.metrics = [
                BrierScore(),
                LogScore(),
                MAE(),
                RMSE(),
                MAPE(),
                CalibrationScore(),
                SharpnessScore(),
                CRPS()
            ]
        else:
            self.metrics = metrics
    
    def evaluate(
        self,
        predictions: Union[ForecastResult, np.ndarray],
        actuals: Union[ForecastData, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate predictions using all metrics.
        
        Parameters
        ----------
        predictions : ForecastResult or np.ndarray
            Predictions
        actuals : ForecastData or np.ndarray
            Actual values
        
        Returns
        -------
        results : dict
            Dictionary mapping metric names to scores
        """
        results = {}
        for metric in self.metrics:
            try:
                score = metric.evaluate(predictions, actuals)
                results[metric.name] = score
            except Exception as e:
                warnings.warn(f"Error computing {metric.name}: {e}")
                results[metric.name] = np.nan
        
        return results
