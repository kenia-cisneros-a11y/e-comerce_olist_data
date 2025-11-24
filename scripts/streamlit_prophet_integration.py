"""
Streamlit Integration for Prophet Forecasting
==============================================
Code to integrate Prophet forecasting as tab7 in the Detailed Performance section.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from prophet_model import ProphetForecaster
from prophet_visualization import ProphetVisualizer
import plotly.graph_objects as go


def add_prophet_tab(df: pd.DataFrame, date_column: str, value_column: str):
    """
    Add Prophet forecasting tab to Streamlit app.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with time series data
    date_column : str
        Name of the date column
    value_column : str
        Name of the value column to forecast
    """
    
    # Initialize session state for model persistence
    if 'prophet_model' not in st.session_state:
        st.session_state.prophet_model = None
    if 'forecast_data' not in st.session_state:
        st.session_state.forecast_data = None
    if 'mae_score' not in st.session_state:
        st.session_state.mae_score = None
    
    # Create two columns for controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Model Configuration")
        
        # Training parameters
        st.markdown("**Training Parameters**")
        test_months = st.slider(
            "Test period (months)",
            min_value=1,
            max_value=6,
            value=2,
            help="Number of months to use for model evaluation"
        )
        
        # Model parameters
        with st.expander("Advanced Model Parameters"):
            seasonality_mode = st.selectbox(
                "Seasonality Mode",
                options=['additive', 'multiplicative'],
                index=0
            )
            
            changepoint_prior_scale = st.slider(
                "Changepoint Prior Scale",
                min_value=0.001,
                max_value=0.5,
                value=0.05,
                step=0.001,
                help="Flexibility of the trend (higher = more flexible)"
            )
            
            seasonality_prior_scale = st.slider(
                "Seasonality Prior Scale",
                min_value=0.01,
                max_value=25.0,
                value=10.0,
                step=0.1,
                help="Flexibility of the seasonality"
            )
            
            yearly_seasonality = st.checkbox("Yearly Seasonality", value=True)
            weekly_seasonality = st.checkbox("Weekly Seasonality", value=True)
            daily_seasonality = st.checkbox("Daily Seasonality", value=False)
        
        # Train model button
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            with st.spinner("Training Prophet model..."):
                try:
                    # Initialize forecaster with parameters
                    model_params = {
                        'seasonality_mode': seasonality_mode,
                        'changepoint_prior_scale': changepoint_prior_scale,
                        'seasonality_prior_scale': seasonality_prior_scale,
                        'yearly_seasonality': yearly_seasonality,
                        'weekly_seasonality': weekly_seasonality,
                        'daily_seasonality': daily_seasonality
                    }
                    
                    forecaster = ProphetForecaster(model_params=model_params)
                    
                    # Train the model
                    results = forecaster.train(
                        df=df,
                        date_column=date_column,
                        value_column=value_column,
                        test_months=test_months
                    )
                    
                    # Store in session state
                    st.session_state.prophet_model = forecaster
                    st.session_state.mae_score = results['mae_score']
                    
                    st.success(f"✅ Model trained successfully! MAE: {results['mae_score']:.4f}")
                    
                    # Save results
                    forecaster.save_results('prophet_results.json')
                    
                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")
    
    with col2:
        st.subheader("📊 Forecasting")
        
        # Prediction parameters
        st.markdown("**Prediction Parameters**")
        
        # Custom or default periods
        use_custom_periods = st.checkbox("Use custom forecast period", value=False)
        
        if use_custom_periods:
            # Allow daily, weekly, or monthly forecasts
            forecast_unit = st.selectbox(
                "Forecast Unit",
                options=['Days', 'Weeks', 'Months'],
                index=0
            )
            
            if forecast_unit == 'Days':
                periods = st.number_input(
                    "Number of days to forecast",
                    min_value=1,
                    max_value=365,
                    value=90
                )
                freq = 'D'
            elif forecast_unit == 'Weeks':
                periods = st.number_input(
                    "Number of weeks to forecast",
                    min_value=1,
                    max_value=52,
                    value=12
                )
                periods = periods * 7  # Convert to days
                freq = 'D'
            else:  # Months
                periods = st.number_input(
                    "Number of months to forecast",
                    min_value=1,
                    max_value=24,
                    value=3
                )
                periods = periods * 30  # Approximate conversion to days
                freq = 'D'
        else:
            # Default: 3 months
            periods = 90
            freq = 'D'
            st.info("📅 Default: Forecasting for 3 months (90 days)")
        
        include_history = st.checkbox("Include historical data in forecast", value=True)
        
        # Generate forecast button
        if st.button("📈 Generate Forecast", type="secondary", use_container_width=True):
            if st.session_state.prophet_model is not None:
                with st.spinner("Generating forecast..."):
                    try:
                        forecaster = st.session_state.prophet_model
                        forecast = forecaster.predict(
                            periods=int(periods),
                            freq=freq,
                            include_history=include_history
                        )
                        st.session_state.forecast_data = forecast
                        st.success("✅ Forecast generated successfully!")
                    except Exception as e:
                        st.error(f"❌ Error generating forecast: {str(e)}")
            else:
                st.warning("⚠️ Please train the model first!")
        
        # Model export/import
        st.markdown("**Model Management**")
        col_export, col_import = st.columns(2)
        
        with col_export:
            if st.button("💾 Save Model", use_container_width=True):
                if st.session_state.prophet_model is not None:
                    try:
                        st.session_state.prophet_model.save_model('prophet_model.json')
                        st.success("Model saved!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.warning("No model to save!")
        
        with col_import:
            uploaded_model = st.file_uploader(
                "Load Model",
                type=['json'],
                key="prophet_model_upload"
            )
            if uploaded_model is not None:
                try:
                    model_data = json.loads(uploaded_model.read())
                    forecaster = ProphetForecaster()
                    # Save temporary file and load
                    with open('temp_model.json', 'w') as f:
                        json.dump(model_data, f)
                    forecaster.load_model('temp_model.json')
                    st.session_state.prophet_model = forecaster
                    st.success("Model loaded!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
    
    # Divider
    st.divider()
    
    # Visualization section
    if st.session_state.prophet_model is not None:
        st.subheader("📊 Visualizations")
        
        visualizer = ProphetVisualizer()
        forecaster = st.session_state.prophet_model
        
        # Create tabs for different visualizations
        viz_tabs = st.tabs([
            "📈 Forecast Plot", 
            "🔍 Components", 
            "📊 Performance Metrics",
            "📉 Residuals Analysis"
        ])
        
        with viz_tabs[0]:
            if st.session_state.forecast_data is not None:
                try:
                    fig = visualizer.plot_forecast(
                        model=forecaster.model,
                        forecast=st.session_state.forecast_data,
                        train_data=forecaster.train_data,
                        test_data=forecaster.test_data,
                        periods_to_highlight=periods if not include_history else None
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display key metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MAE Score", f"{st.session_state.mae_score:.4f}")
                    with col2:
                        st.metric("Training Samples", len(forecaster.train_data))
                    with col3:
                        st.metric("Test Samples", len(forecaster.test_data))
                    
                except Exception as e:
                    st.error(f"Error creating forecast plot: {str(e)}")
            else:
                st.info("Generate a forecast to see the plot")
        
        with viz_tabs[1]:
            if st.session_state.forecast_data is not None:
                try:
                    fig = visualizer.plot_components(
                        model=forecaster.model,
                        forecast=st.session_state.forecast_data
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating components plot: {str(e)}")
            else:
                st.info("Generate a forecast to see components")
        
        with viz_tabs[2]:
            if forecaster.test_data is not None:
                try:
                    # Get predictions for test data
                    test_forecast = forecaster.model.predict(forecaster.test_data[['ds']])
                    
                    fig = visualizer.plot_performance_metrics(
                        train_data=forecaster.train_data,
                        test_data=forecaster.test_data,
                        predictions=test_forecast,
                        mae_score=st.session_state.mae_score
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional metrics
                    st.subheader("📊 Detailed Metrics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Calculate additional metrics
                        from sklearn.metrics import mean_squared_error, r2_score
                        
                        merged = forecaster.test_data.merge(
                            test_forecast[['ds', 'yhat']], 
                            on='ds', 
                            how='inner'
                        )
                        
                        mse = mean_squared_error(merged['y'], merged['yhat'])
                        rmse = np.sqrt(mse)
                        r2 = r2_score(merged['y'], merged['yhat'])
                        mape = np.mean(np.abs((merged['y'] - merged['yhat']) / merged['y'])) * 100
                        
                        st.metric("MAE", f"{st.session_state.mae_score:.4f}")
                        st.metric("RMSE", f"{rmse:.4f}")
                    
                    with col2:
                        st.metric("R² Score", f"{r2:.4f}")
                        st.metric("MAPE (%)", f"{mape:.2f}")
                    
                except Exception as e:
                    st.error(f"Error creating performance plot: {str(e)}")
            else:
                st.info("Train the model to see performance metrics")
        
        with viz_tabs[3]:
            if forecaster.test_data is not None:
                try:
                    # Calculate residuals
                    test_forecast = forecaster.model.predict(forecaster.test_data[['ds']])
                    merged = forecaster.test_data.merge(
                        test_forecast[['ds', 'yhat']], 
                        on='ds', 
                        how='inner'
                    )
                    residuals = merged['y'] - merged['yhat']
                    
                    fig = visualizer.plot_residuals_analysis(
                        residuals=residuals,
                        dates=merged['ds']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error creating residuals plot: {str(e)}")
            else:
                st.info("Train the model to see residuals analysis")
    
    else:
        st.info("👆 Configure and train the model to see visualizations")


# Example integration into existing Streamlit app
def integrate_prophet_into_app():
    """
    Example of how to integrate Prophet tab into existing Streamlit app
    with Detailed Performance section.
    """
    
    st.title("Performance Analysis Dashboard")
    
    # Existing tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Tab 1",
        "Tab 2", 
        "Tab 3",
        "Tab 4",
        "Tab 5",
        "Tab 6",
        "📈 Prophet Forecast"  # New tab7
    ])
    
    # ... existing code for tabs 1-6 ...
    
    # Tab 7: Prophet Forecasting
    with tab7:
        # Load your data
        # This is an example - replace with your actual data loading
        try:
            # Example: Load data from uploaded file or database
            df = pd.read_csv('your_data.csv')  # Replace with actual data source
            date_column = 'date'  # Replace with your date column name
            value_column = 'value'  # Replace with your value column name
            
            # Add Prophet forecasting functionality
            add_prophet_tab(df, date_column, value_column)
            
        except FileNotFoundError:
            st.warning("Please upload or connect to your data source")
            
            # Alternative: File upload option
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                
                # Let user select columns
                cols = st.columns(2)
                with cols[0]:
                    date_column = st.selectbox("Select date column", df.columns)
                with cols[1]:
                    value_column = st.selectbox("Select value column", df.columns)
                
                add_prophet_tab(df, date_column, value_column)


if __name__ == "__main__":
    # Run example integration
    integrate_prophet_into_app()