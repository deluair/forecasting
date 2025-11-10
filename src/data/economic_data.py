"""
Economic data loaders for common data sources.

Supports World Bank, IMF, FRED, and other economic data APIs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import warnings

try:
    import wbdata
    WBDATA_AVAILABLE = True
except ImportError:
    WBDATA_AVAILABLE = False
    warnings.warn("wbdata not available. World Bank data loading will not work.")

from ..core import ForecastData


class WorldBankLoader:
    """
    Loader for World Bank economic data.
    
    Requires wbdata package: pip install wbdata
    """
    
    def __init__(self):
        """Initialize World Bank loader."""
        if not WBDATA_AVAILABLE:
            raise ImportError("wbdata package required. Install with: pip install wbdata")
    
    def load_country_data(
        self,
        country_code: str,
        indicators: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load country economic data from World Bank.
        
        Parameters
        ----------
        country_code : str
            ISO country code (e.g., 'BGD' for Bangladesh, 'USA' for United States)
        indicators : list of str
            World Bank indicator codes (e.g., 'NY.GDP.MKTP.KD.ZG' for GDP growth)
        start_date : str, optional
            Start date (e.g., '2010')
        end_date : str, optional
            End date (e.g., '2024')
        
        Returns
        -------
        df : pd.DataFrame
            Economic data with dates as index
        """
        import wbdata as wb
        
        data = wb.get_dataframe(
            {ind: ind for ind in indicators},
            country=country_code,
            data_date=(start_date, end_date) if start_date and end_date else None,
            convert_date=True
        )
        
        return data
    
    def get_indicator_info(self, indicator_code: str) -> Dict:
        """
        Get information about a World Bank indicator.
        
        Parameters
        ----------
        indicator_code : str
            Indicator code
        
        Returns
        -------
        info : dict
            Indicator information
        """
        import wbdata as wb
        
        indicators = wb.get_indicator(indicator_code)
        return indicators


class FREDLoader:
    """
    Loader for Federal Reserve Economic Data (FRED).
    
    Requires fredapi package: pip install fredapi
    Requires FRED API key (free): https://fred.stlouisfed.org/docs/api/api_key.html
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED loader.
        
        Parameters
        ----------
        api_key : str, optional
            FRED API key. If None, looks for FRED_API_KEY environment variable
        """
        try:
            from fredapi import Fred
            self.fred = Fred(api_key=api_key)
            self.api_available = True
        except ImportError:
            self.api_available = False
            warnings.warn("fredapi not available. Install with: pip install fredapi")
    
    def load_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.Series:
        """
        Load economic series from FRED.
        
        Parameters
        ----------
        series_id : str
            FRED series ID (e.g., 'GDP' for US GDP)
        start_date : str, optional
            Start date (YYYY-MM-DD format)
        end_date : str, optional
            End date (YYYY-MM-DD format)
        
        Returns
        -------
        series : pd.Series
            Economic time series
        """
        if not self.api_available:
            raise ImportError("fredapi required. Install with: pip install fredapi")
        
        series = self.fred.get_series(
            series_id,
            observation_start=start_date,
            observation_end=end_date
        )
        
        return series


class EconomicDataConverter:
    """Convert economic data to ForecastData format."""
    
    @staticmethod
    def from_world_bank(
        df: pd.DataFrame,
        indicator_name: str,
        metadata: Optional[Dict] = None
    ) -> ForecastData:
        """
        Convert World Bank DataFrame to ForecastData.
        
        Parameters
        ----------
        df : pd.DataFrame
            World Bank data
        indicator_name : str
            Name of the indicator column
        metadata : dict, optional
            Additional metadata
        
        Returns
        -------
        ForecastData
            Forecast data object
        """
        if indicator_name not in df.columns:
            raise ValueError(f"Indicator '{indicator_name}' not found in data")
        
        timestamps = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
        values = df[indicator_name].dropna().values
        
        # Align timestamps with values
        valid_idx = ~pd.isna(df[indicator_name])
        timestamps = timestamps[valid_idx]
        
        meta = metadata or {}
        meta['source'] = 'World Bank'
        meta['indicator'] = indicator_name
        
        return ForecastData(timestamps, values, metadata=meta)
    
    @staticmethod
    def from_fred(
        series: pd.Series,
        metadata: Optional[Dict] = None
    ) -> ForecastData:
        """
        Convert FRED Series to ForecastData.
        
        Parameters
        ----------
        series : pd.Series
            FRED economic series
        metadata : dict, optional
            Additional metadata
        
        Returns
        -------
        ForecastData
            Forecast data object
        """
        timestamps = series.index
        values = series.dropna().values
        
        valid_idx = ~pd.isna(series)
        timestamps = timestamps[valid_idx]
        
        meta = metadata or {}
        meta['source'] = 'FRED'
        
        return ForecastData(timestamps, values, metadata=meta)

