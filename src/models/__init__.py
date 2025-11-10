"""
Forecasting models.
"""

from .ensemble import EnsembleForecaster, WeightedEnsemble
from .time_series import ARIMAForecaster, ProphetForecaster
from .bayesian import BayesianForecaster
from .ml import MLForecaster

try:
    from .econometric import VARForecaster, VECMForecaster, StateSpaceForecaster
    ECONOMETRIC_AVAILABLE = True
except ImportError:
    ECONOMETRIC_AVAILABLE = False

__all__ = [
    'EnsembleForecaster',
    'WeightedEnsemble',
    'ARIMAForecaster',
    'ProphetForecaster',
    'BayesianForecaster',
    'MLForecaster',
]

if ECONOMETRIC_AVAILABLE:
    __all__.extend(['VARForecaster', 'VECMForecaster', 'StateSpaceForecaster'])
