"""
Main package initialization.
"""

__version__ = "0.1.0"
__author__ = "Forecasting Team"

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

