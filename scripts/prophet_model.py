"""
Prophet Model Module
====================
A module for time series forecasting using Prophet with JSON serialization support.
Includes training, testing, evaluation and prediction capabilities.
"""

import json
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProphetForecaster:
    """
    A wrapper class for Prophet model with enhanced functionality for:
    - Training with train/test split
    - Model serialization to JSON
    - Performance evaluation with MAE
    - Future predictions
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Prophet forecaster.
        
        Parameters
        ----------
        model_params : dict, optional
            Prophet model parameters (seasonality, changepoint_prior_scale, etc.)
        """
        self.model_params = model_params or {}
        self.model = None
        self.train_data = None
        self.test_data = None
        self.mae_score = None
        self.training_history = None
        
    def prepare_data(self, df: pd.DataFrame, date_column: str, value_column: str) -> pd.DataFrame:
        """
        Prepare data for Prophet (requires columns 'ds' and 'y').
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        date_column : str
            Name of the date column
        value_column : str
            Name of the value column to forecast
            
        Returns
        -------
        pd.DataFrame
            Formatted dataframe with 'ds' and 'y' columns
        """
        prophet_df = df[[date_column, value_column]].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
        
        # Handle missing values
        prophet_df = prophet_df.dropna()
        
        return prophet_df
    
    def split_train_test(self, df: pd.DataFrame, test_months: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets based on time.
        
        Parameters
        ----------
        df : pd.DataFrame
            Prepared dataframe with 'ds' and 'y' columns
        test_months : int, default=2
            Number of months to use for testing
            
        Returns
        -------
        tuple
            (train_df, test_df)
        """
        df['ds'] = pd.to_datetime(df['ds'])
        max_date = df['ds'].max()
        
        # Calculate split date
        split_date = max_date - pd.DateOffset(months=test_months)
        
        train_df = df[df['ds'] <= split_date].copy()
        test_df = df[df['ds'] > split_date].copy()
        
        logger.info(f"Train period: {train_df['ds'].min()} to {train_df['ds'].max()}")
        logger.info(f"Test period: {test_df['ds'].min()} to {test_df['ds'].max()}")
        logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
        
        return train_df, test_df
    
    def train(self, df: pd.DataFrame, date_column: str, value_column: str, 
              test_months: int = 2, **kwargs) -> Dict[str, Any]:
        """
        Train the Prophet model using only the training set.
        
        Parameters
        ----------
        df : pd.DataFrame
            Complete dataset
        date_column : str
            Name of the date column
        value_column : str
            Name of the value column to forecast
        test_months : int, default=2
            Number of months to use for testing
        **kwargs
            Additional Prophet parameters
            
        Returns
        -------
        dict
            Training results including MAE on test set
        """
        # Prepare data
        prophet_df = self.prepare_data(df, date_column, value_column)
        
        # Split train/test
        self.train_data, self.test_data = self.split_train_test(prophet_df, test_months)
        
        # Initialize and train model (only on train data)
        model_params = {**self.model_params, **kwargs}
        self.model = Prophet(**model_params)
        
        logger.info("Training Prophet model...")
        self.model.fit(self.train_data)
        
        # Evaluate on test set
        test_forecast = self.model.predict(self.test_data[['ds']])
        self.mae_score = mean_absolute_error(self.test_data['y'], test_forecast['yhat'])
        
        # Store training history
        self.training_history = {
            'train_start': str(self.train_data['ds'].min()),
            'train_end': str(self.train_data['ds'].max()),
            'test_start': str(self.test_data['ds'].min()),
            'test_end': str(self.test_data['ds'].max()),
            'train_samples': len(self.train_data),
            'test_samples': len(self.test_data),
            'mae_score': float(self.mae_score),
            'model_params': model_params,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Training completed. MAE on test set: {self.mae_score:.4f}")
        
        return self.training_history
    
    def predict(self, periods: int = 90, freq: str = 'D', 
                include_history: bool = True) -> pd.DataFrame:
        """
        Make future predictions.
        
        Parameters
        ----------
        periods : int, default=90
            Number of periods to forecast (default is 3 months = ~90 days)
        freq : str, default='D'
            Frequency of predictions ('D' for daily, 'M' for monthly, etc.)
        include_history : bool, default=True
            Whether to include historical data in the forecast
            
        Returns
        -------
        pd.DataFrame
            Forecast dataframe with predictions and confidence intervals
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq, 
                                                  include_history=include_history)
        
        # Make predictions
        forecast = self.model.predict(future)
        
        # Add actual values where available
        if include_history and self.train_data is not None:
            # Merge with actual training data
            actual_data = pd.concat([self.train_data, self.test_data], ignore_index=True)
            forecast = forecast.merge(actual_data, on='ds', how='left')
        
        return forecast
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model parameters to a JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to save the JSON file
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Serialize model to JSON
        model_json = model_to_json(self.model)
        
        # Add metadata
        save_data = {
            'model': json.loads(model_json),
            'training_history': self.training_history,
            'mae_score': float(self.mae_score) if self.mae_score else None,
            'saved_at': datetime.now().isoformat()
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load model parameters from a JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to the JSON file
        """
        with open(filepath, 'r') as f:
            save_data = json.load(f)
        
        # Restore model
        model_json = json.dumps(save_data['model'])
        self.model = model_from_json(model_json)
        
        # Restore metadata
        self.training_history = save_data.get('training_history')
        self.mae_score = save_data.get('mae_score')
        
        logger.info(f"Model loaded from {filepath}")
    
    def save_results(self, filepath: str, test_predictions: Optional[pd.DataFrame] = None) -> None:
        """
        Save training results and test evaluation to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the results file
        test_predictions : pd.DataFrame, optional
            Test predictions to include in results
        """
        if self.training_history is None:
            raise ValueError("No training results to save.")
        
        results = {
            'training_history': self.training_history,
            'mae_score': float(self.mae_score) if self.mae_score else None,
            'test_metrics': {
                'mae': float(self.mae_score) if self.mae_score else None,
                'test_samples': len(self.test_data) if self.test_data is not None else 0
            }
        }
        
        # Add test predictions if provided
        if test_predictions is not None:
            results['test_predictions'] = test_predictions.to_dict('records')
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def get_model_components(self) -> pd.DataFrame:
        """
        Get model components (trend, seasonality, etc.).
        
        Returns
        -------
        pd.DataFrame
            DataFrame with model components
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get forecast for historical data
        forecast = self.model.predict(self.train_data[['ds']])
        
        return forecast
    
    def cross_validate(self, initial: str = '730 days', period: str = '180 days', 
                       horizon: str = '60 days') -> pd.DataFrame:
        """
        Perform cross-validation to assess model performance.
        
        Parameters
        ----------
        initial : str, default='730 days'
            Initial training period
        period : str, default='180 days'
            Period between cutoff dates
        horizon : str, default='60 days'
            Forecast horizon
            
        Returns
        -------
        pd.DataFrame
            Cross-validation results
        """
        from prophet.diagnostics import cross_validation, performance_metrics
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Perform cross-validation
        cv_results = cross_validation(self.model, initial=initial, 
                                     period=period, horizon=horizon)
        
        # Calculate performance metrics
        metrics = performance_metrics(cv_results)
        
        return metrics


def main():
    """
    Example usage of the ProphetForecaster class.
    """
    # Generate sample data
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
    values = np.random.randn(len(dates)).cumsum() + 100 + \
              10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    
    df = pd.DataFrame({
        'date': dates,
        'value': values
    })
    
    # Initialize and train model
    forecaster = ProphetForecaster(model_params={'seasonality_mode': 'multiplicative'})
    
    # Train with 2 months test period (default)
    results = forecaster.train(df, 'date', 'value', test_months=2)
    print(f"Training Results: {results}")
    
    # Make predictions for 3 months (default)
    forecast = forecaster.predict(periods=90)
    print(f"Forecast shape: {forecast.shape}")
    
    # Save model and results
    forecaster.save_model('prophet_model.json')
    forecaster.save_results('training_results.json')
    
    # Load model
    new_forecaster = ProphetForecaster()
    new_forecaster.load_model('prophet_model.json')
    
    # Make new predictions
    new_forecast = new_forecaster.predict(periods=30)
    print(f"New forecast shape: {new_forecast.shape}")


if __name__ == "__main__":
    main()