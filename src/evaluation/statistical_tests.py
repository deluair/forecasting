"""
Statistical tests for forecast evaluation and validation.

Includes tests for forecast accuracy comparison, residual analysis,
and time series properties.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
from scipy import stats
import warnings

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.stattools import jarque_bera
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Some statistical tests will not work.")


class DieboldMarianoTest:
    """
    Diebold-Mariano test for comparing forecast accuracy.
    
    Tests the null hypothesis that two forecasts have equal accuracy.
    
    References:
    - Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
    """
    
    def __init__(self, h: int = 1):
        """
        Initialize Diebold-Mariano test.
        
        Parameters
        ----------
        h : int
            Forecast horizon (for small-sample correction)
        """
        self.h = h
    
    def test(
        self,
        forecast1: np.ndarray,
        forecast2: np.ndarray,
        actuals: np.ndarray,
        loss_function: str = 'squared'
    ) -> Tuple[float, float, float]:
        """
        Perform Diebold-Mariano test.
        
        Parameters
        ----------
        forecast1 : np.ndarray
            First forecast
        forecast2 : np.ndarray
            Second forecast
        actuals : np.ndarray
            Actual values
        loss_function : str
            Loss function: 'squared', 'absolute', or 'percentage'
        
        Returns
        -------
        dm_statistic : float
            DM test statistic
        p_value : float
            P-value for two-sided test
        critical_value : float
            Critical value at 5% significance level
        """
        # Calculate loss differential
        if loss_function == 'squared':
            loss1 = (forecast1 - actuals) ** 2
            loss2 = (forecast2 - actuals) ** 2
        elif loss_function == 'absolute':
            loss1 = np.abs(forecast1 - actuals)
            loss2 = np.abs(forecast2 - actuals)
        elif loss_function == 'percentage':
            loss1 = np.abs((forecast1 - actuals) / (actuals + 1e-10))
            loss2 = np.abs((forecast2 - actuals) / (actuals + 1e-10))
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
        
        d = loss1 - loss2
        
        # Calculate DM statistic
        d_mean = np.mean(d)
        d_var = np.var(d, ddof=1)
        
        # Small-sample correction
        n = len(d)
        if n > self.h:
            # HAC (Heteroskedasticity and Autocorrelation Consistent) variance
            # Simplified version - full implementation would use Newey-West
            dm_stat = d_mean / np.sqrt(d_var / n)
        else:
            dm_stat = d_mean / np.sqrt(d_var / n)
        
        # P-value (two-sided test)
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
        
        # Critical value at 5%
        critical_value = stats.norm.ppf(0.975)
        
        return dm_stat, p_value, critical_value


class LjungBoxTest:
    """
    Ljung-Box test for residual autocorrelation.
    
    Tests the null hypothesis that residuals are independently distributed.
    
    References:
    - Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in time series models.
    """
    
    def __init__(self, lags: Optional[int] = None):
        """
        Initialize Ljung-Box test.
        
        Parameters
        ----------
        lags : int, optional
            Number of lags to test. If None, uses min(10, n/5)
        """
        self.lags = lags
    
    def test(self, residuals: np.ndarray) -> Tuple[float, float]:
        """
        Perform Ljung-Box test.
        
        Parameters
        ----------
        residuals : np.ndarray
            Residuals from forecast model
        
        Returns
        -------
        q_statistic : float
            Ljung-Box Q statistic
        p_value : float
            P-value
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for LjungBoxTest")
        
        if self.lags is None:
            self.lags = min(10, len(residuals) // 5)
        
        result = acorr_ljungbox(residuals, lags=self.lags, return_df=False)
        q_stat = result[0][-1]  # Last lag statistic
        p_value = result[1][-1]  # Last lag p-value
        
        return q_stat, p_value


class AugmentedDickeyFullerTest:
    """
    Augmented Dickey-Fuller test for unit roots.
    
    Tests the null hypothesis that a time series has a unit root (non-stationary).
    
    References:
    - Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series.
    """
    
    def __init__(self, maxlag: Optional[int] = None):
        """
        Initialize ADF test.
        
        Parameters
        ----------
        maxlag : int, optional
            Maximum lag length. If None, uses automatic selection
        """
        self.maxlag = maxlag
    
    def test(
        self,
        series: np.ndarray,
        regression: str = 'c'
    ) -> Tuple[float, float, dict]:
        """
        Perform Augmented Dickey-Fuller test.
        
        Parameters
        ----------
        series : np.ndarray
            Time series to test
        regression : str
            Type of regression: 'c' (constant), 'ct' (constant+trend), 'nc' (no constant)
        
        Returns
        -------
        adf_statistic : float
            ADF test statistic
        p_value : float
            P-value
        critical_values : dict
            Critical values at 1%, 5%, 10%
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for AugmentedDickeyFullerTest")
        
        result = adfuller(series, maxlag=self.maxlag, regression=regression, autolag='AIC')
        
        adf_stat = result[0]
        p_value = result[1]
        critical_values = {
            '1%': result[4]['1%'],
            '5%': result[4]['5%'],
            '10%': result[4]['10%']
        }
        
        return adf_stat, p_value, critical_values


class JarqueBeraTest:
    """
    Jarque-Bera test for normality.
    
    Tests the null hypothesis that residuals are normally distributed.
    
    References:
    - Jarque, C. M., & Bera, A. K. (1980). Efficient tests for normality, homoscedasticity.
    """
    
    def test(self, residuals: np.ndarray) -> Tuple[float, float]:
        """
        Perform Jarque-Bera test.
        
        Parameters
        ----------
        residuals : np.ndarray
            Residuals to test
        
        Returns
        -------
        jb_statistic : float
            Jarque-Bera statistic
        p_value : float
            P-value
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for JarqueBeraTest")
        
        jb_stat, p_value = jarque_bera(residuals)
        return jb_stat, p_value


class ForecastValidation:
    """
    Comprehensive forecast validation suite.
    
    Performs multiple statistical tests on forecast residuals.
    """
    
    def __init__(self):
        """Initialize validation suite."""
        self.dm_test = DieboldMarianoTest()
        self.lb_test = LjungBoxTest()
        self.adf_test = AugmentedDickeyFullerTest()
        self.jb_test = JarqueBeraTest()
    
    def validate(
        self,
        forecast: np.ndarray,
        actuals: np.ndarray,
        baseline_forecast: Optional[np.ndarray] = None
    ) -> dict:
        """
        Perform comprehensive forecast validation.
        
        Parameters
        ----------
        forecast : np.ndarray
            Forecast values
        actuals : np.ndarray
            Actual values
        baseline_forecast : np.ndarray, optional
            Baseline forecast for comparison (e.g., naive forecast)
        
        Returns
        -------
        results : dict
            Dictionary containing all test results
        """
        residuals = actuals - forecast
        
        results = {
            'residuals': {
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals)),
                'skewness': float(stats.skew(residuals)),
                'kurtosis': float(stats.kurtosis(residuals))
            },
            'tests': {}
        }
        
        # Ljung-Box test
        try:
            q_stat, p_value = self.lb_test.test(residuals)
            results['tests']['ljung_box'] = {
                'statistic': float(q_stat),
                'p_value': float(p_value),
                'null_hypothesis': 'No autocorrelation',
                'reject_null': p_value < 0.05
            }
        except Exception as e:
            results['tests']['ljung_box'] = {'error': str(e)}
        
        # Jarque-Bera test
        try:
            jb_stat, p_value = self.jb_test.test(residuals)
            results['tests']['jarque_bera'] = {
                'statistic': float(jb_stat),
                'p_value': float(p_value),
                'null_hypothesis': 'Normally distributed',
                'reject_null': p_value < 0.05
            }
        except Exception as e:
            results['tests']['jarque_bera'] = {'error': str(e)}
        
        # ADF test on residuals
        try:
            adf_stat, p_value, crit_values = self.adf_test.test(residuals)
            results['tests']['adf'] = {
                'statistic': float(adf_stat),
                'p_value': float(p_value),
                'critical_values': crit_values,
                'null_hypothesis': 'Unit root present',
                'reject_null': p_value < 0.05
            }
        except Exception as e:
            results['tests']['adf'] = {'error': str(e)}
        
        # Diebold-Mariano test (if baseline provided)
        if baseline_forecast is not None:
            try:
                dm_stat, p_value, crit_value = self.dm_test.test(
                    forecast, baseline_forecast, actuals
                )
                results['tests']['diebold_mariano'] = {
                    'statistic': float(dm_stat),
                    'p_value': float(p_value),
                    'critical_value': float(crit_value),
                    'null_hypothesis': 'Equal forecast accuracy',
                    'reject_null': p_value < 0.05
                }
            except Exception as e:
                results['tests']['diebold_mariano'] = {'error': str(e)}
        
        return results

