"""
Example: Prediction competition workflow (Metaculus/GJ Open)
"""

import pandas as pd
import numpy as np

from src.data import MetaculusLoader, PredictionDataConverter
from src.models import EnsembleForecaster, ARIMAForecaster, MLForecaster
from src.evaluation import BrierScore, LogScore, CalibrationScore, MetricSuite
from src.utils import CalibrationTool
from src.visualization import ForecastPlotter


def process_competition_questions(loader, questions_df):
    """Process questions from competition data."""
    results = []
    
    for _, row in questions_df.iterrows():
        question_id = row['question_id']
        print(f"\nProcessing question: {question_id}")
        
        # Check if resolved
        if not row.get('is_resolved', False):
            print(f"  Question not yet resolved, skipping...")
            continue
        
        # Get resolution
        resolution_value = row['resolution_value']
        resolution_date = row['resolution_date']
        
        # For binary questions
        if resolution_value in [0, 1, True, False]:
            # Convert to ForecastData format
            # In practice, you'd have historical prediction data
            # For this example, we'll simulate
            
            # Simulate historical predictions (in practice, load from API/data)
            dates = pd.date_range(
                end=resolution_date,
                periods=30,
                freq='D'
            )
            # Simulate predictions trending toward resolution
            true_prob = float(resolution_value)
            predictions = np.random.beta(
                alpha=10 * true_prob + 1,
                beta=10 * (1 - true_prob) + 1,
                size=30
            )
            
            # Create forecast data
            train_data = PredictionDataConverter.from_binary_question(
                question_id=question_id,
                resolution_date=resolution_date,
                resolution_value=resolution_value,
                prediction_history=pd.DataFrame({
                    'timestamp': dates,
                    'probability': predictions
                })
            )
            
            # Split data
            train, test = train_data.split(0.8)
            
            # Fit forecaster
            forecaster = MLForecaster(model_type='random_forest', n_lags=5)
            forecaster.fit(train)
            
            # Predict
            forecast = forecaster.predict(horizon=len(test.values))
            
            # Evaluate
            brier = BrierScore()
            log_score = LogScore()
            calibration = CalibrationScore()
            
            brier_score = brier.evaluate(forecast, test)
            log_score_val = log_score.evaluate(forecast, test)
            cal_score = calibration.evaluate(forecast, test)
            
            results.append({
                'question_id': question_id,
                'brier_score': brier_score,
                'log_score': log_score_val,
                'calibration_score': cal_score,
                'resolution': resolution_value
            })
            
            print(f"  Brier Score: {brier_score:.4f}")
            print(f"  Log Score: {log_score_val:.4f}")
            print(f"  Calibration Score: {cal_score:.4f}")
    
    return pd.DataFrame(results)


def main():
    """Main competition workflow."""
    print("Loading competition data...")
    
    # Example: Load Metaculus data
    # In practice, you'd have actual data files
    # loader = MetaculusLoader("data/metaculus_questions.csv")
    # df = loader.load()
    
    # For demonstration, create sample data
    print("Creating sample competition data...")
    sample_data = {
        'question_id': ['Q1', 'Q2', 'Q3'],
        'question_text': ['Will X happen?', 'Will Y happen?', 'Will Z happen?'],
        'resolution_date': pd.to_datetime(['2024-01-15', '2024-02-01', '2024-03-01']),
        'resolution_value': [1, 0, 1],
        'is_resolved': [True, True, True],
        'community_median': [0.6, 0.3, 0.7]
    }
    df = pd.DataFrame(sample_data)
    
    # Process questions
    results = process_competition_questions(None, df)
    
    # Summary statistics
    print("\n" + "="*50)
    print("COMPETITION RESULTS SUMMARY")
    print("="*50)
    print(f"\nTotal questions processed: {len(results)}")
    print(f"Mean Brier Score: {results['brier_score'].mean():.4f}")
    print(f"Mean Log Score: {results['log_score'].mean():.4f}")
    print(f"Mean Calibration Score: {results['calibration_score'].mean():.4f}")
    
    # Save results
    results.to_csv('examples/competition_results.csv', index=False)
    print("\nResults saved to examples/competition_results.csv")


if __name__ == '__main__':
    main()

