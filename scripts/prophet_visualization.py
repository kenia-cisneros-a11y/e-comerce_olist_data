"""
Prophet Visualization Module
============================
Module for creating interactive visualizations of Prophet forecasts using Plotly.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from prophet.plot import plot_plotly, plot_components_plotly
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ProphetVisualizer:
    """
    A class for creating interactive visualizations of Prophet forecasts.
    """
    
    def __init__(self):
        """Initialize the visualizer with default settings."""
        self.default_colors = {
            'actual': '#1f77b4',
            'forecast': '#ff7f0e',
            'trend': '#2ca02c',
            'confidence': 'rgba(255, 127, 14, 0.2)',
            'test': '#d62728'
        }
    
    def plot_forecast(self, model, forecast: pd.DataFrame, 
                     train_data: Optional[pd.DataFrame] = None,
                     test_data: Optional[pd.DataFrame] = None,
                     periods_to_highlight: Optional[int] = None) -> go.Figure:
        """
        Create an interactive forecast plot using Plotly.
        
        Parameters
        ----------
        model : Prophet
            Trained Prophet model
        forecast : pd.DataFrame
            Forecast dataframe from Prophet
        train_data : pd.DataFrame, optional
            Training data with 'ds' and 'y' columns
        test_data : pd.DataFrame, optional
            Test data with 'ds' and 'y' columns
        periods_to_highlight : int, optional
            Number of future periods to highlight
            
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        # Use Prophet's built-in plotly function as base
        fig = plot_plotly(model, forecast)
        
        # Customize the layout
        fig.update_layout(
            title={
                'text': 'Prophet Forecast',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white',
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add test data if provided
        if test_data is not None:
            fig.add_trace(go.Scatter(
                x=test_data['ds'],
                y=test_data['y'],
                mode='markers',
                name='Test Data',
                marker=dict(
                    color=self.default_colors['test'],
                    size=8,
                    symbol='diamond'
                )
            ))
        
        # Highlight future predictions if specified
        if periods_to_highlight is not None:
            future_start = forecast['ds'].max() - pd.Timedelta(days=periods_to_highlight)
            future_forecast = forecast[forecast['ds'] > future_start]
            
            if not future_forecast.empty:
                fig.add_vrect(
                    x0=future_start,
                    x1=forecast['ds'].max(),
                    fillcolor="rgba(255, 0, 0, 0.05)",
                    layer="below",
                    line_width=0,
                )
        
        return fig
    
    def plot_components(self, model, forecast: pd.DataFrame) -> go.Figure:
        """
        Create an interactive components plot using Plotly.
        
        Parameters
        ----------
        model : Prophet
            Trained Prophet model
        forecast : pd.DataFrame
            Forecast dataframe from Prophet
            
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        # Use Prophet's built-in plotly function
        fig = plot_components_plotly(model, forecast)
        
        # Customize the layout
        fig.update_layout(
            title={
                'text': 'Forecast Components',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            template='plotly_white',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def plot_performance_metrics(self, train_data: pd.DataFrame, 
                                test_data: pd.DataFrame,
                                predictions: pd.DataFrame,
                                mae_score: float) -> go.Figure:
        """
        Create a performance visualization comparing actual vs predicted values.
        
        Parameters
        ----------
        train_data : pd.DataFrame
            Training data with 'ds' and 'y' columns
        test_data : pd.DataFrame
            Test data with 'ds' and 'y' columns
        predictions : pd.DataFrame
            Predictions for test period
        mae_score : float
            Mean Absolute Error score
            
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Actual vs Predicted', 'Prediction Errors'),
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4]
        )
        
        # Plot 1: Actual vs Predicted
        # Training data
        fig.add_trace(
            go.Scatter(
                x=train_data['ds'],
                y=train_data['y'],
                mode='lines',
                name='Training Data',
                line=dict(color=self.default_colors['actual'])
            ),
            row=1, col=1
        )
        
        # Test data actual
        fig.add_trace(
            go.Scatter(
                x=test_data['ds'],
                y=test_data['y'],
                mode='markers+lines',
                name='Test Actual',
                line=dict(color=self.default_colors['test']),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Test predictions
        fig.add_trace(
            go.Scatter(
                x=predictions['ds'],
                y=predictions['yhat'],
                mode='markers+lines',
                name='Test Predicted',
                line=dict(color=self.default_colors['forecast'], dash='dash'),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Add confidence intervals if available
        if 'yhat_lower' in predictions.columns and 'yhat_upper' in predictions.columns:
            fig.add_trace(
                go.Scatter(
                    x=predictions['ds'].tolist() + predictions['ds'].tolist()[::-1],
                    y=predictions['yhat_upper'].tolist() + predictions['yhat_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor=self.default_colors['confidence'],
                    line=dict(width=0),
                    name='Confidence Interval',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Plot 2: Prediction Errors
        merged_data = test_data.merge(predictions[['ds', 'yhat']], on='ds', how='inner')
        errors = merged_data['y'] - merged_data['yhat']
        
        fig.add_trace(
            go.Scatter(
                x=merged_data['ds'],
                y=errors,
                mode='lines+markers',
                name='Prediction Error',
                line=dict(color='purple'),
                marker=dict(size=5)
            ),
            row=2, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, row=2, col=1, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'Model Performance (MAE: {mae_score:.4f})',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            template='plotly_white',
            height=800,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Error", row=2, col=1)
        
        return fig
    
    def plot_residuals_analysis(self, residuals: pd.Series, dates: pd.Series) -> go.Figure:
        """
        Create a comprehensive residuals analysis plot.
        
        Parameters
        ----------
        residuals : pd.Series
            Residuals (actual - predicted)
        dates : pd.Series
            Corresponding dates
            
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Residuals Over Time', 'Residuals Distribution',
                          'Q-Q Plot', 'ACF Plot'),
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # 1. Residuals over time
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color='blue', size=5, opacity=0.6)
            ),
            row=1, col=1
        )
        fig.add_hline(y=0, row=1, col=1, line_dash="dash", line_color="red", opacity=0.5)
        
        # 2. Histogram of residuals
        fig.add_trace(
            go.Histogram(
                x=residuals,
                nbinsx=30,
                name='Distribution',
                marker_color='green',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # 3. Q-Q Plot
        from scipy import stats
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
        sample_quantiles = np.sort(residuals)
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                name='Q-Q',
                marker=dict(color='purple', size=5)
            ),
            row=2, col=1
        )
        
        # Add diagonal line for Q-Q plot
        min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Normal Line',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=1
        )
        
        # 4. ACF Plot (simplified)
        from statsmodels.stats.stattools import acf
        acf_values = acf(residuals, nlags=40)
        lags = np.arange(len(acf_values))
        
        fig.add_trace(
            go.Bar(
                x=lags,
                y=acf_values,
                name='ACF',
                marker_color='orange'
            ),
            row=2, col=2
        )
        
        # Add significance bounds
        significance_level = 1.96 / np.sqrt(len(residuals))
        fig.add_hline(y=significance_level, row=2, col=2, 
                     line_dash="dash", line_color="blue", opacity=0.5)
        fig.add_hline(y=-significance_level, row=2, col=2, 
                     line_dash="dash", line_color="blue", opacity=0.5)
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Residuals Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            template='plotly_white',
            height=700,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Residual", row=1, col=1)
        fig.update_xaxes(title_text="Residual", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)
        fig.update_xaxes(title_text="Lag", row=2, col=2)
        fig.update_yaxes(title_text="ACF", row=2, col=2)
        
        return fig