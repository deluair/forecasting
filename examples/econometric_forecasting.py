"""
Advanced econometric forecasting example for PhD-level economic analysis.

Demonstrates VAR models, impulse response analysis, and comprehensive
statistical validation suitable for academic research.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.core import ForecastData
from src.models.econometric import VARForecaster
from src.models import ARIMAForecaster, EnsembleForecaster
from src.evaluation import MetricSuite
from src.evaluation.statistical_tests import (
    DieboldMarianoTest,
    LjungBoxTest,
    ForecastValidation
)
from src.visualization import ForecastPlotter


def generate_multivariate_economic_data():
    """
    Generate realistic multivariate economic data.
    
    Simulates GDP growth, inflation, and interest rate with realistic
    correlations and dynamics.
    """
    np.random.seed(42)
    n_obs = 100
    dates = pd.date_range(start='2010-01-01', periods=n_obs, freq='Q')
    
    # Generate correlated economic variables
    # GDP Growth (dependent on lagged values and other variables)
    gdp_growth = np.zeros(n_obs)
    inflation = np.zeros(n_obs)
    interest_rate = np.zeros(n_obs)
    
    # Initial values
    gdp_growth[0] = 6.0
    inflation[0] = 6.0
    interest_rate[0] = 5.0
    
    # Generate with VAR-like dynamics
    for t in range(1, n_obs):
        # GDP growth: depends on lagged GDP and inflation
        gdp_growth[t] = (
            0.7 * gdp_growth[t-1] +
            -0.3 * inflation[t-1] +
            np.random.normal(0, 0.5)
        )
        
        # Inflation: depends on lagged inflation and interest rate
        inflation[t] = (
            0.6 * inflation[t-1] +
            -0.2 * interest_rate[t-1] +
            np.random.normal(0, 0.4)
        )
        
        # Interest rate: depends on lagged interest rate and inflation
        interest_rate[t] = (
            0.8 * interest_rate[t-1] +
            0.3 * inflation[t-1] +
            np.random.normal(0, 0.3)
        )
    
    # Add trend and bounds
    gdp_growth = np.clip(gdp_growth + np.linspace(0, 1, n_obs), 4.0, 8.0)
    inflation = np.clip(inflation + np.linspace(0, -0.5, n_obs), 3.0, 8.0)
    interest_rate = np.clip(interest_rate, 3.0, 10.0)
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'gdp_growth': gdp_growth,
        'inflation': inflation,
        'interest_rate': interest_rate
    })
    
    return data


def main():
    """Advanced econometric forecasting demonstration."""
    print("="*70)
    print("ADVANCED ECONOMETRIC FORECASTING")
    print("="*70)
    print("\nThis example demonstrates:")
    print("  1. Vector Autoregression (VAR) modeling")
    print("  2. Impulse Response Analysis")
    print("  3. Statistical validation and testing")
    print("  4. Forecast comparison using Diebold-Mariano test")
    print("\n" + "="*70)
    
    # Load/generate data
    print("\n1. Loading Economic Data...")
    economic_data = generate_multivariate_economic_data()
    print(f"   Period: {economic_data['date'].min().date()} to {economic_data['date'].max().date()}")
    print(f"   Variables: GDP Growth, Inflation, Interest Rate")
    print(f"   Observations: {len(economic_data)}")
    
    # Prepare multivariate data for VAR
    var_data = economic_data[['gdp_growth', 'inflation', 'interest_rate']].values
    
    # Create ForecastData for GDP growth (univariate forecasting)
    gdp_data = ForecastData(
        economic_data['date'],
        economic_data['gdp_growth'].values,
        metadata={'variable': 'GDP Growth', 'unit': '%'}
    )
    
    # Split data
    train_data, test_data = gdp_data.split(0.8)
    print(f"\n2. Data Splitting:")
    print(f"   Training: {len(train_data.values)} observations")
    print(f"   Testing: {len(test_data.values)} observations")
    
    # Fit VAR model on multivariate data
    print(f"\n3. Fitting Vector Autoregression (VAR) Model...")
    try:
        var_train_data = ForecastData(
            economic_data['date'][:len(train_data.values)],
            var_data[:len(train_data.values)],
            metadata={'variables': ['GDP Growth', 'Inflation', 'Interest Rate']}
        )
        
        var_model = VARForecaster(maxlags=4, ic='aic')
        var_model.fit(var_train_data)
        
        print(f"   Selected lags (AIC): {var_model.selected_lags.aic if var_model.selected_lags else 'N/A'}")
        
        # Impulse Response Analysis
        print(f"\n4. Impulse Response Analysis...")
        irf = var_model.impulse_response(periods=10)
        print(f"   IRF shape: {irf.shape}")
        print(f"   Shows response of variables to one-unit shocks")
        
    except Exception as e:
        print(f"   VAR model error: {e}")
        var_model = None
    
    # Fit univariate models for comparison
    print(f"\n5. Fitting Univariate Models...")
    models = {}
    
    # ARIMA
    try:
        arima = ARIMAForecaster(auto=True, name='ARIMA')
        arima.fit(train_data)
        arima_forecast = arima.predict(horizon=len(test_data.values))
        models['ARIMA'] = {'model': arima, 'forecast': arima_forecast}
        print(f"   ARIMA: Fitted successfully")
    except Exception as e:
        print(f"   ARIMA: Error - {e}")
    
    # Statistical Validation
    print(f"\n6. Statistical Validation...")
    if 'ARIMA' in models:
        forecast = models['ARIMA']['forecast']
        residuals = test_data.values - forecast.point_forecast
        
        # Ljung-Box test
        lb_test = LjungBoxTest(lags=10)
        try:
            q_stat, p_value = lb_test.test(residuals)
            print(f"   Ljung-Box Test:")
            print(f"     Q-statistic: {q_stat:.4f}")
            print(f"     P-value: {p_value:.4f}")
            print(f"     Interpretation: {'Residuals are independent' if p_value > 0.05 else 'Residuals show autocorrelation'}")
        except Exception as e:
            print(f"   Ljung-Box Test: Error - {e}")
        
        # Comprehensive validation
        validation = ForecastValidation()
        try:
            # Compare with naive forecast (random walk)
            naive_forecast = np.full_like(test_data.values, train_data.values[-1])
            
            validation_results = validation.validate(
                forecast.point_forecast,
                test_data.values,
                baseline_forecast=naive_forecast
            )
            
            print(f"\n   Forecast Validation Results:")
            print(f"     Residual Mean: {validation_results['residuals']['mean']:.4f}")
            print(f"     Residual Std: {validation_results['residuals']['std']:.4f}")
            
            if 'diebold_mariano' in validation_results['tests']:
                dm = validation_results['tests']['diebold_mariano']
                print(f"     Diebold-Mariano Test:")
                print(f"       Statistic: {dm.get('statistic', 'N/A'):.4f}")
                print(f"       P-value: {dm.get('p_value', 'N/A'):.4f}")
                print(f"       Interpretation: {'ARIMA significantly better' if dm.get('p_value', 1) < 0.05 else 'No significant difference'}")
        
        except Exception as e:
            print(f"   Validation error: {e}")
    
    # Evaluation Metrics
    print(f"\n7. Evaluation Metrics...")
    if 'ARIMA' in models:
        metrics = MetricSuite()
        results = metrics.evaluate(models['ARIMA']['forecast'], test_data)
        
        print(f"   Proper Scoring Rules:")
        print(f"     Brier Score: {results.get('BrierScore', 'N/A'):.4f}")
        print(f"     Log Score: {results.get('LogScore', 'N/A'):.4f}")
        print(f"     CRPS: {results.get('CRPS', 'N/A'):.4f}")
        print(f"\n   Point Forecast Metrics:")
        print(f"     MAE: {results.get('MAE', 'N/A'):.4f}")
        print(f"     RMSE: {results.get('RMSE', 'N/A'):.4f}")
        print(f"     MAPE: {results.get('MAPE', 'N/A'):.2f}%")
    
    # Visualization
    print(f"\n8. Generating Visualizations...")
    os.makedirs('results', exist_ok=True)
    
    if 'ARIMA' in models:
        plotter = ForecastPlotter(figsize=(14, 8))
        fig = plotter.plot_forecast(
            models['ARIMA']['forecast'],
            actuals=test_data,
            train_data=train_data,
            title='GDP Growth Forecast with Statistical Validation',
            show_uncertainty=True
        )
        fig.savefig('results/econometric_forecast.png', dpi=150, bbox_inches='tight')
        print(f"   Saved: results/econometric_forecast.png")
        plt.close(fig)
    
    print(f"\n" + "="*70)
    print("ECONOMETRIC ANALYSIS COMPLETE")
    print("="*70)
    print("\nKey Features Demonstrated:")
    print("  ✅ Vector Autoregression (VAR) modeling")
    print("  ✅ Impulse Response Functions")
    print("  ✅ Statistical tests (Ljung-Box, Diebold-Mariano)")
    print("  ✅ Proper scoring rules for forecast evaluation")
    print("  ✅ Comprehensive residual analysis")
    print("\nThis framework is suitable for:")
    print("  • Academic research in econometrics")
    print("  • Central bank forecasting")
    print("  • Economic policy analysis")
    print("  • Publication-quality forecasts")


if __name__ == '__main__':
    import os
    main()

