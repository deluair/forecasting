"""
Evaluation metrics for forecasting.
"""

from .metrics import (
    BaseMetric,
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

try:
    from .statistical_tests import (
        DieboldMarianoTest,
        LjungBoxTest,
        AugmentedDickeyFullerTest,
        JarqueBeraTest,
        ForecastValidation
    )
    STATISTICAL_TESTS_AVAILABLE = True
except ImportError:
    STATISTICAL_TESTS_AVAILABLE = False

__all__ = [
    'BaseMetric',
    'BrierScore',
    'LogScore',
    'MAE',
    'RMSE',
    'MAPE',
    'CalibrationScore',
    'SharpnessScore',
    'CRPS',
    'MetricSuite',
]

if STATISTICAL_TESTS_AVAILABLE:
    __all__.extend([
        'DieboldMarianoTest',
        'LjungBoxTest',
        'AugmentedDickeyFullerTest',
        'JarqueBeraTest',
        'ForecastValidation'
    ])
