"""
Advanced Forecasting Framework for Prediction Competitions.

A comprehensive, production-ready forecasting framework designed for prediction
competitions like Metaculus and GJ Open. Provides advanced forecasting models,
evaluation metrics, calibration tools, and visualization capabilities.
"""

__version__ = "0.1.0"
__author__ = "deluair"
__license__ = "MIT"
__email__ = "deluair@users.noreply.github.com"

from .core import BaseForecaster, ForecastData, ForecastResult
from .models import (
    EnsembleForecaster,
    WeightedEnsemble,
    ARIMAForecaster,
    ProphetForecaster,
    BayesianForecaster,
    MLForecaster
)
from .data import (
    MetaculusLoader,
    GJOpenLoader,
    PredictionDataConverter
)
from .evaluation import (
    BrierScore,
    LogScore,
    MAE,
    RMSE,
    MAPE,
    CalibrationScore,
    SharpnessScore,
    CRPS,
    MetricSuite
)
from .visualization import ForecastPlotter, ForecastReport
from .utils import CalibrationTool, UncertaintyQuantifier, FeatureEngineering

__all__ = [
    # Core
    'BaseForecaster',
    'ForecastData',
    'ForecastResult',
    # Models
    'EnsembleForecaster',
    'WeightedEnsemble',
    'ARIMAForecaster',
    'ProphetForecaster',
    'BayesianForecaster',
    'MLForecaster',
    # Data
    'MetaculusLoader',
    'GJOpenLoader',
    'PredictionDataConverter',
    # Evaluation
    'BrierScore',
    'LogScore',
    'MAE',
    'RMSE',
    'MAPE',
    'CalibrationScore',
    'SharpnessScore',
    'CRPS',
    'MetricSuite',
    # Visualization
    'ForecastPlotter',
    'ForecastReport',
    # Utils
    'CalibrationTool',
    'UncertaintyQuantifier',
    'FeatureEngineering',
]

