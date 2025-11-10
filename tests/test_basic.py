import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from src.core import ForecastData, ForecastResult, BaseForecaster
from src.models import ARIMAForecaster, EnsembleForecaster
from src.evaluation import BrierScore, MAE, MetricSuite
from src.data import MetaculusLoader


class TestForecastData:
    """Tests for ForecastData."""
    
    def test_creation(self):
        """Test ForecastData creation."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        values = np.random.randn(10)
        data = ForecastData(dates, values)
        assert len(data.timestamps) == 10
        assert len(data.values) == 10
    
    def test_split(self):
        """Test data splitting."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        values = np.random.randn(100)
        data = ForecastData(dates, values)
        
        train, test = data.split(0.8)
        assert len(train.values) == 80
        assert len(test.values) == 20


class TestForecastResult:
    """Tests for ForecastResult."""
    
    def test_creation(self):
        """Test ForecastResult creation."""
        pred = np.array([1, 2, 3])
        result = ForecastResult(pred)
        assert len(result.point_forecast) == 3
    
    def test_uncertainty_intervals(self):
        """Test uncertainty interval extraction."""
        pred = np.array([1, 2, 3])
        lower = np.array([0.5, 1.5, 2.5])
        upper = np.array([1.5, 2.5, 3.5])
        result = ForecastResult(pred, lower_bound=lower, upper_bound=upper)
        
        l, u = result.get_uncertainty_intervals()
        assert np.allclose(l, lower)
        assert np.allclose(u, upper)


class TestMetrics:
    """Tests for evaluation metrics."""
    
    def test_brier_score(self):
        """Test Brier Score calculation."""
        predictions = np.array([0.7, 0.8, 0.3])
        actuals = np.array([1, 1, 0])
        
        brier = BrierScore()
        score = brier.evaluate(predictions, actuals)
        assert 0 <= score <= 1
    
    def test_mae(self):
        """Test MAE calculation."""
        predictions = np.array([1, 2, 3])
        actuals = np.array([1.5, 2.5, 3.5])
        
        mae = MAE()
        score = mae.evaluate(predictions, actuals)
        assert score == 0.5
    
    def test_metric_suite(self):
        """Test MetricSuite."""
        predictions = np.array([1, 2, 3])
        actuals = np.array([1.5, 2.5, 3.5])
        
        suite = MetricSuite()
        results = suite.evaluate(predictions, actuals)
        assert 'MAE' in results
        assert 'RMSE' in results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

