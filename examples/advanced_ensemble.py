"""
Example: Advanced ensemble with calibration
"""

import numpy as np
import pandas as pd
from datetime import datetime

from src.core import ForecastData
from src.models import EnsembleForecaster, ARIMAForecaster, ProphetForecaster, MLForecaster
from src.evaluation import MetricSuite, CalibrationScore
from src.utils import CalibrationTool
from src.visualization import ForecastPlotter


def main():
    """Demonstrate advanced ensemble with calibration."""
    print("Advanced Ensemble Forecasting Example")
    print("="*50)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
    trend = np.linspace(0, 15, 300)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(300) / 365.25)
    noise = np.random.normal(0, 1.5, 300)
    values = trend + seasonal + noise
    
    data = ForecastData(dates, values)
    train_data, test_data = data.split(0.75)
    
    print(f"   Training: {len(train_data.values)} points")
    print(f"   Test: {len(test_data.values)} points")
    
    # Create multiple forecasters
    print("\n2. Creating and fitting forecasters...")
    forecasters = [
        ARIMAForecaster(auto=True, name='ARIMA'),
        ProphetForecaster(yearly_seasonality=True, name='Prophet'),
        MLForecaster(model_type='random_forest', n_lags=10, name='RandomForest'),
        MLForecaster(model_type='gradient_boosting', n_lags=10, name='GradientBoosting')
    ]
    
    for forecaster in forecasters:
        print(f"   Fitting {forecaster.name}...")
        forecaster.fit(train_data)
    
    # Create ensemble with learned weights
    print("\n3. Creating weighted ensemble...")
    from src.models import WeightedEnsemble
    
    ensemble = WeightedEnsemble(
        forecasters=forecasters,
        optimization_method='inverse_variance'
    )
    ensemble.fit(train_data)
    print(f"   Ensemble weights: {ensemble.weights}")
    
    # Make predictions
    print("\n4. Generating forecasts...")
    horizon = len(test_data.values)
    ensemble_pred = ensemble.predict(horizon=horizon)
    
    # Evaluate before calibration
    print("\n5. Evaluating forecasts...")
    metrics = MetricSuite()
    metrics_before = metrics.evaluate(ensemble_pred, test_data)
    
    print("\nMetrics (before calibration):")
    for metric, value in metrics_before.items():
        print(f"   {metric}: {value:.4f}")
    
    # Apply calibration
    print("\n6. Applying calibration...")
    # For binary outcomes, use calibration
    # For continuous, we'll demonstrate with a simplified approach
    
    # Evaluate calibration
    calibration = CalibrationScore()
    cal_score_before = calibration.evaluate(ensemble_pred, test_data)
    print(f"   Calibration score (before): {cal_score_before:.4f}")
    
    # Visualize
    print("\n7. Generating visualizations...")
    plotter = ForecastPlotter()
    
    fig = plotter.plot_forecast(
        ensemble_pred,
        actuals=test_data,
        train_data=train_data,
        title='Advanced Ensemble Forecast'
    )
    fig.savefig('examples/advanced_ensemble.png', dpi=150, bbox_inches='tight')
    print("   Saved to examples/advanced_ensemble.png")
    
    # Plot calibration curve
    if len(test_data.values) > 20:  # Need enough points for calibration
        # Convert to probabilities for calibration plot
        # (simplified - in practice, you'd have actual probabilities)
        pred_probs = np.clip(
            (ensemble_pred.point_forecast - ensemble_pred.point_forecast.min()) /
            (ensemble_pred.point_forecast.max() - ensemble_pred.point_forecast.min() + 1e-10),
            0, 1
        )
        actual_binary = (test_data.values > np.median(test_data.values)).astype(int)
        
        fig_cal = plotter.plot_calibration(pred_probs, actual_binary)
        fig_cal.savefig('examples/calibration_curve.png', dpi=150, bbox_inches='tight')
        print("   Saved to examples/calibration_curve.png")
    
    print("\n" + "="*50)
    print("Example completed successfully!")
    print("="*50)


if __name__ == '__main__':
    main()

