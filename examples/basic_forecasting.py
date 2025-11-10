"""
Example: Basic forecasting workflow
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.core import ForecastData
from src.models import ARIMAForecaster, ProphetForecaster, EnsembleForecaster
from src.evaluation import MetricSuite
from src.visualization import ForecastPlotter


def generate_sample_data(n=100):
    """Generate sample time series data."""
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    
    # Generate synthetic data with trend and seasonality
    trend = np.linspace(0, 10, n)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    noise = np.random.normal(0, 1, n)
    values = trend + seasonal + noise
    
    return ForecastData(dates, values, metadata={'source': 'synthetic'})


def main():
    """Main example workflow."""
    print("Generating sample data...")
    data = generate_sample_data(n=200)
    
    # Split into train and test
    train_data, test_data = data.split(0.8)
    print(f"Training data: {len(train_data.values)} points")
    print(f"Test data: {len(test_data.values)} points")
    
    # Create forecasters
    print("\nFitting forecasters...")
    arima = ARIMAForecaster(auto=True)
    arima.fit(train_data)
    
    prophet = ProphetForecaster(yearly_seasonality=True, weekly_seasonality=True)
    prophet.fit(train_data)
    
    # Create ensemble
    ensemble = EnsembleForecaster(
        forecasters=[arima, prophet],
        method='weighted_average',
        weights='equal'
    )
    ensemble.fit(train_data)
    
    # Make predictions
    print("\nGenerating forecasts...")
    horizon = len(test_data.values)
    
    arima_pred = arima.predict(horizon=horizon)
    prophet_pred = prophet.predict(horizon=horizon)
    ensemble_pred = ensemble.predict(horizon=horizon)
    
    # Evaluate
    print("\nEvaluating forecasts...")
    metrics = MetricSuite()
    
    arima_metrics = metrics.evaluate(arima_pred, test_data)
    prophet_metrics = metrics.evaluate(prophet_pred, test_data)
    ensemble_metrics = metrics.evaluate(ensemble_pred, test_data)
    
    print("\nARIMA Metrics:")
    for metric, value in arima_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nProphet Metrics:")
    for metric, value in prophet_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nEnsemble Metrics:")
    for metric, value in ensemble_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Visualize
    print("\nGenerating plots...")
    plotter = ForecastPlotter()
    
    # Plot ensemble forecast
    fig = plotter.plot_forecast(
        ensemble_pred,
        actuals=test_data,
        train_data=train_data,
        title='Ensemble Forecast'
    )
    fig.savefig('examples/ensemble_forecast.png', dpi=150, bbox_inches='tight')
    print("Saved plot to examples/ensemble_forecast.png")
    
    # Plot residuals
    fig_residuals = plotter.plot_residuals(ensemble_pred, test_data)
    fig_residuals.savefig('examples/residuals.png', dpi=150, bbox_inches='tight')
    print("Saved plot to examples/residuals.png")


if __name__ == '__main__':
    main()

