"""
Data handlers for prediction competitions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime

from ..core import ForecastData


class CompetitionDataLoader:
    """Base class for competition data loaders."""
    
    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize competition data loader.
        
        Parameters
        ----------
        data_path : str or Path
            Path to the competition data file
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
    
    def load(self) -> pd.DataFrame:
        """Load competition data."""
        raise NotImplementedError("Subclasses must implement load()")
    
    def parse_questions(self, df: pd.DataFrame) -> List[Dict]:
        """Parse questions from loaded data."""
        raise NotImplementedError("Subclasses must implement parse_questions()")


class MetaculusLoader(CompetitionDataLoader):
    """
    Loader for Metaculus prediction competition data.
    
    Expected format:
    - CSV with columns: question_id, question_text, resolution_date, 
      resolution_value, community_median, community_mean, etc.
    """
    
    def load(self) -> pd.DataFrame:
        """Load Metaculus data from CSV."""
        df = pd.read_csv(self.data_path)
        
        # Standardize column names
        column_mapping = {
            'id': 'question_id',
            'question': 'question_text',
            'resolution': 'resolution_value',
            'resolved': 'is_resolved',
        }
        df = df.rename(columns=column_mapping)
        
        # Parse dates
        date_columns = ['resolution_date', 'created_date', 'publish_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def parse_questions(self, df: pd.DataFrame) -> List[Dict]:
        """
        Parse Metaculus questions into standardized format.
        
        Parameters
        ----------
        df : pd.DataFrame
            Loaded Metaculus data
        
        Returns
        -------
        questions : list of dict
            Parsed questions with standardized fields
        """
        questions = []
        
        for _, row in df.iterrows():
            question = {
                'question_id': row.get('question_id', ''),
                'question_text': row.get('question_text', ''),
                'resolution_date': row.get('resolution_date'),
                'resolution_value': row.get('resolution_value'),
                'is_resolved': row.get('is_resolved', False),
                'community_median': row.get('community_median'),
                'community_mean': row.get('community_mean'),
                'created_date': row.get('created_date'),
                'publish_date': row.get('publish_date'),
                'raw_data': row.to_dict()
            }
            questions.append(question)
        
        return questions
    
    def get_binary_questions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract binary (yes/no) questions."""
        # Binary questions typically have resolution_value in [0, 1] or boolean
        binary_mask = (
            df['resolution_value'].isin([0, 1, True, False]) |
            df['resolution_value'].isna()
        )
        return df[binary_mask].copy()
    
    def get_continuous_questions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract continuous (numeric) questions."""
        binary_mask = df['resolution_value'].isin([0, 1, True, False])
        return df[~binary_mask].copy()


class GJOpenLoader(CompetitionDataLoader):
    """
    Loader for GJ Open prediction competition data.
    
    Expected format:
    - CSV with columns: question_id, question, resolution, 
      community_prediction, etc.
    """
    
    def load(self) -> pd.DataFrame:
        """Load GJ Open data from CSV."""
        df = pd.read_csv(self.data_path)
        
        # Standardize column names
        column_mapping = {
            'id': 'question_id',
            'question': 'question_text',
            'resolution': 'resolution_value',
            'resolved': 'is_resolved',
        }
        df = df.rename(columns=column_mapping)
        
        # Parse dates
        date_columns = ['resolution_date', 'created_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def parse_questions(self, df: pd.DataFrame) -> List[Dict]:
        """
        Parse GJ Open questions into standardized format.
        
        Parameters
        ----------
        df : pd.DataFrame
            Loaded GJ Open data
        
        Returns
        -------
        questions : list of dict
            Parsed questions with standardized fields
        """
        questions = []
        
        for _, row in df.iterrows():
            question = {
                'question_id': row.get('question_id', ''),
                'question_text': row.get('question_text', ''),
                'resolution_date': row.get('resolution_date'),
                'resolution_value': row.get('resolution_value'),
                'is_resolved': row.get('is_resolved', False),
                'community_prediction': row.get('community_prediction'),
                'created_date': row.get('created_date'),
                'raw_data': row.to_dict()
            }
            questions.append(question)
        
        return questions


class PredictionDataConverter:
    """Convert competition data to ForecastData format."""
    
    @staticmethod
    def from_time_series(
        timestamps: Union[List, np.ndarray, pd.DatetimeIndex],
        values: Union[List, np.ndarray],
        metadata: Optional[Dict] = None
    ) -> ForecastData:
        """
        Create ForecastData from time series.
        
        Parameters
        ----------
        timestamps : array-like
            Timestamps
        values : array-like
            Values
        metadata : dict, optional
            Additional metadata
        
        Returns
        -------
        ForecastData
            Forecast data object
        """
        return ForecastData(timestamps, values, metadata)
    
    @staticmethod
    def from_question_history(
        question_id: str,
        history_df: pd.DataFrame,
        value_column: str = 'value',
        timestamp_column: str = 'timestamp'
    ) -> ForecastData:
        """
        Create ForecastData from question history.
        
        Parameters
        ----------
        question_id : str
            Question identifier
        history_df : pd.DataFrame
            Historical data for the question
        value_column : str
            Column name for values
        timestamp_column : str
            Column name for timestamps
        
        Returns
        -------
        ForecastData
            Forecast data object
        """
        timestamps = pd.to_datetime(history_df[timestamp_column])
        values = history_df[value_column].values
        
        metadata = {
            'question_id': question_id,
            'n_observations': len(values)
        }
        
        return ForecastData(timestamps, values, metadata)
    
    @staticmethod
    def from_binary_question(
        question_id: str,
        resolution_date: datetime,
        resolution_value: Union[int, float, bool],
        prediction_history: Optional[pd.DataFrame] = None
    ) -> ForecastData:
        """
        Create ForecastData for binary question.
        
        Parameters
        ----------
        question_id : str
            Question identifier
        resolution_date : datetime
            When the question was resolved
        resolution_value : int, float, or bool
            Resolution value (0/1 or True/False)
        prediction_history : pd.DataFrame, optional
            Historical predictions
        
        Returns
        -------
        ForecastData
            Forecast data object
        """
        # Convert resolution to binary
        resolution = float(resolution_value)
        if resolution not in [0.0, 1.0]:
            resolution = 1.0 if resolution else 0.0
        
        if prediction_history is not None:
            timestamps = pd.to_datetime(prediction_history['timestamp'])
            values = prediction_history['probability'].values
        else:
            # Single data point
            timestamps = pd.to_datetime([resolution_date])
            values = np.array([resolution])
        
        metadata = {
            'question_id': question_id,
            'resolution_date': resolution_date,
            'resolution_value': resolution,
            'is_binary': True
        }
        
        return ForecastData(timestamps, values, metadata)
