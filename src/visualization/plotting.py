"""
Visualization and reporting tools for forecasts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, List, Dict
import warnings

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Some visualization features will be limited.")

from ..core import ForecastResult, ForecastData


class ForecastPlotter:
    """Plotting utilities for forecasts."""
    
    def __init__(self, style: str = 'seaborn', figsize: tuple = (12, 6)):
        """
        Initialize plotter.
        
        Parameters
        ----------
        style : str
            Matplotlib style
        figsize : tuple
            Figure size
        """
        self.style = style
        self.figsize = figsize
        try:
            plt.style.use(style)
        except OSError:
            # Fallback to default style if requested style not available
            plt.style.use('default')
    
    def plot_forecast(
        self,
        forecast: ForecastResult,
        actuals: Optional[ForecastData] = None,
        train_data: Optional[ForecastData] = None,
        show_uncertainty: bool = True,
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot forecast with optional actuals and training data.
        
        Parameters
        ----------
        forecast : ForecastResult
            Forecast to plot
        actuals : ForecastData, optional
            Actual values for comparison
        train_data : ForecastData, optional
            Training data to show
        show_uncertainty : bool
            Whether to show uncertainty intervals
        title : str, optional
            Plot title
        ax : plt.Axes, optional
            Matplotlib axes to plot on
        **kwargs
            Additional plotting parameters
        
        Returns
        -------
        fig : plt.Figure
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        # Plot training data
        if train_data is not None:
            ax.plot(
                train_data.timestamps,
                train_data.values,
                'o-',
                label='Training Data',
                color='gray',
                alpha=0.7,
                markersize=4
            )
        
        # Plot forecast
        timestamps = forecast.timestamps if forecast.timestamps is not None else None
        if timestamps is None:
            timestamps = np.arange(len(forecast.point_forecast))
        
        ax.plot(
            timestamps,
            forecast.point_forecast,
            'o-',
            label='Forecast',
            color='blue',
            linewidth=2,
            markersize=6
        )
        
        # Plot uncertainty intervals
        if show_uncertainty:
            if forecast.lower_bound is not None and forecast.upper_bound is not None:
                ax.fill_between(
                    timestamps,
                    forecast.lower_bound,
                    forecast.upper_bound,
                    alpha=0.3,
                    color='blue',
                    label='Uncertainty Interval'
                )
            elif forecast.quantiles:
                # Use quantiles if available
                quantiles = sorted(forecast.quantiles.keys())
                if len(quantiles) >= 2:
                    lower_q = quantiles[0]
                    upper_q = quantiles[-1]
                    ax.fill_between(
                        timestamps,
                        forecast.quantiles[lower_q],
                        forecast.quantiles[upper_q],
                        alpha=0.3,
                        color='blue',
                        label=f'Uncertainty ({lower_q:.0%}-{upper_q:.0%})'
                    )
        
        # Plot actuals
        if actuals is not None:
            ax.plot(
                actuals.timestamps,
                actuals.values,
                's-',
                label='Actual',
                color='red',
                linewidth=2,
                markersize=6
            )
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title or 'Forecast', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_residuals(
        self,
        forecast: ForecastResult,
        actuals: ForecastData,
        title: Optional[str] = None,
        figsize: tuple = (12, 4)
    ) -> plt.Figure:
        """
        Plot forecast residuals.
        
        Parameters
        ----------
        forecast : ForecastResult
            Forecast
        actuals : ForecastData
            Actual values
        title : str, optional
            Plot title
        figsize : tuple
            Figure size
        
        Returns
        -------
        fig : plt.Figure
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Calculate residuals
        residuals = actuals.values - forecast.point_forecast
        
        # Time series of residuals
        axes[0].plot(actuals.timestamps, residuals, 'o-', color='red', alpha=0.7)
        axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Residual')
        axes[0].set_title('Residuals Over Time')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1].hist(residuals, bins=30, color='red', alpha=0.7, edgecolor='black')
        axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1)
        axes[1].set_xlabel('Residual')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution')
        axes[1].grid(True, alpha=0.3)
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_calibration(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        n_bins: int = 10,
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """
        Plot calibration curve (reliability diagram).
        
        Parameters
        ----------
        predictions : np.ndarray
            Predicted probabilities
        actuals : np.ndarray
            Actual binary outcomes
        n_bins : int
            Number of bins
        title : str, optional
            Plot title
        ax : plt.Axes, optional
            Matplotlib axes
        
        Returns
        -------
        fig : plt.Figure
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure
        
        # Bin predictions
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        observed_frequencies = []
        predicted_frequencies = []
        counts = []
        
        for i in range(len(bin_boundaries) - 1):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find predictions in this bin
            in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
            count = in_bin.sum()
            counts.append(count)
            
            if count > 0:
                observed_freq = actuals[in_bin].mean()
                predicted_freq = predictions[in_bin].mean()
            else:
                observed_freq = np.nan
                predicted_freq = np.nan
            
            observed_frequencies.append(observed_freq)
            predicted_frequencies.append(predicted_freq)
        
        # Plot
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        ax.plot(predicted_frequencies, observed_frequencies, 'o-', label='Forecast', linewidth=2, markersize=8)
        
        # Add count annotations
        for i, (x, y, count) in enumerate(zip(predicted_frequencies, observed_frequencies, counts)):
            if not np.isnan(x) and not np.isnan(y) and count > 0:
                ax.annotate(str(count), (x, y), fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Observed Frequency', fontsize=12)
        ax.set_title(title or 'Calibration Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        return fig


class ForecastReport:
    """Generate comprehensive forecast reports."""
    
    def __init__(self, plotter: Optional[ForecastPlotter] = None):
        """
        Initialize report generator.
        
        Parameters
        ----------
        plotter : ForecastPlotter, optional
            Plotter instance
        """
        self.plotter = plotter or ForecastPlotter()
    
    def generate_report(
        self,
        forecast: ForecastResult,
        actuals: Optional[ForecastData] = None,
        train_data: Optional[ForecastData] = None,
        metrics: Optional[Dict[str, float]] = None,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Generate comprehensive forecast report.
        
        Parameters
        ----------
        forecast : ForecastResult
            Forecast
        actuals : ForecastData, optional
            Actual values
        train_data : ForecastData, optional
            Training data
        metrics : dict, optional
            Evaluation metrics
        save_path : str, optional
            Path to save report
        
        Returns
        -------
        report : dict
            Report dictionary
        """
        report = {
            'forecast_summary': self._summarize_forecast(forecast),
            'metrics': metrics or {},
            'plots': {}
        }
        
        # Generate plots
        if train_data is not None or actuals is not None:
            fig = self.plotter.plot_forecast(
                forecast,
                actuals=actuals,
                train_data=train_data
            )
            report['plots']['forecast'] = fig
        
        if actuals is not None:
            fig_residuals = self.plotter.plot_residuals(forecast, actuals)
            report['plots']['residuals'] = fig_residuals
        
        # Save if requested
        if save_path:
            self._save_report(report, save_path)
        
        return report
    
    def _summarize_forecast(self, forecast: ForecastResult) -> Dict:
        """Summarize forecast statistics."""
        summary = {
            'n_predictions': len(forecast.point_forecast),
            'mean': float(np.mean(forecast.point_forecast)),
            'std': float(np.std(forecast.point_forecast)),
            'min': float(np.min(forecast.point_forecast)),
            'max': float(np.max(forecast.point_forecast))
        }
        
        if forecast.lower_bound is not None and forecast.upper_bound is not None:
            summary['uncertainty_width'] = float(np.mean(forecast.upper_bound - forecast.lower_bound))
        
        return summary
    
    def _save_report(self, report: Dict, save_path: str):
        """Save report to file."""
        # Save plots
        for plot_name, fig in report['plots'].items():
            plot_path = save_path.replace('.html', f'_{plot_name}.png')
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        # Save summary as JSON
        import json
        summary_path = save_path.replace('.html', '_summary.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'forecast_summary': report['forecast_summary'],
                'metrics': report['metrics']
            }, f, indent=2)
