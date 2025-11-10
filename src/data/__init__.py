"""
Data handlers for prediction competitions.
"""

from .competition_loader import (
    CompetitionDataLoader,
    MetaculusLoader,
    GJOpenLoader,
    PredictionDataConverter
)

try:
    from .economic_data import (
        WorldBankLoader,
        FREDLoader,
        EconomicDataConverter
    )
    ECONOMIC_DATA_AVAILABLE = True
except ImportError:
    ECONOMIC_DATA_AVAILABLE = False

__all__ = [
    'CompetitionDataLoader',
    'MetaculusLoader',
    'GJOpenLoader',
    'PredictionDataConverter',
]

if ECONOMIC_DATA_AVAILABLE:
    __all__.extend(['WorldBankLoader', 'FREDLoader', 'EconomicDataConverter'])
