"""
Cohort Analysis Module for E-commerce Dashboard
Compatible with existing key_metrics.py and Streamlit visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


def cohort_analysis_dashboard(orders_df, order_payments_df, customers_df=None, 
                             date_range=None, cohort_period='month', 
                             retention_type='customers', max_periods=12):
    """
    Complete cohort analysis function designed for Streamlit dashboards.
    
    Parameters:
    -----------
    orders_df : DataFrame - Orders data with timestamps
    order_payments_df : DataFrame - Payment data for revenue cohorts
    customers_df : DataFrame - Customer data (optional)
    date_range : tuple - (start_date, end_date) for filtering
    cohort_period : str - 'month', 'week', or 'quarter' for cohort grouping
    retention_type : str - 'customers' or 'revenue' for retention metric
    max_periods : int - Maximum number of periods to analyze
    
    Returns:
    --------
    dict containing:
        - retention_matrix: Retention percentages by cohort and period
        - cohort_sizes: Initial cohort sizes
        - metrics: Key performance metrics
        - visualizations: Matplotlib figures ready for Streamlit
    """
    
    # Prepare data
    df = orders_df.copy()
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    
    # Apply date filter
    if date_range and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df = df[(df['order_purchase_timestamp'] >= start_date) & 
                (df['order_purchase_timestamp'] <= end_date)]
    
    if df.empty:
        return None
    
    # Add revenue data if analyzing revenue retention
    if retention_type == 'revenue':
        revenue_by_order = order_payments_df.groupby('order_id')['payment_value'].sum().reset_index()
        df = df.merge(revenue_by_order, on='order_id', how='left')
        df['payment_value'] = df['payment_value'].fillna(0)
    
    # Create cohort identifiers
    if cohort_period == 'month':
        df['order_period'] = df['order_purchase_timestamp'].dt.to_period('M')
        df['cohort'] = df.groupby('customer_id')['order_purchase_timestamp'].transform('min').dt.to_period('M')
    elif cohort_period == 'week':
        df['order_period'] = df['order_purchase_timestamp'].dt.to_period('W')
        df['cohort'] = df.groupby('customer_id')['order_purchase_timestamp'].transform('min').dt.to_period('W')
    else:  # quarter
        df['order_period'] = df['order_purchase_timestamp'].dt.to_period('Q')
        df['cohort'] = df.groupby('customer_id')['order_purchase_timestamp'].transform('min').dt.to_period('Q')
    
    # Calculate periods since first purchase
    df['cohort_index'] = (df['order_period'] - df['cohort']).apply(lambda x: x.n if hasattr(x, 'n') else 0)
    
    # Filter to max periods
    df = df[df['cohort_index'] <= max_periods]
    
    # Create cohort matrix
    if retention_type == 'customers':
        # Customer retention
        cohort_data = df.groupby(['cohort', 'cohort_index'])['customer_id'].nunique().reset_index()
        cohort_pivot = cohort_data.pivot(index='cohort', columns='cohort_index', values='customer_id')
    else:
        # Revenue retention
        cohort_data = df.groupby(['cohort', 'cohort_index'])['payment_value'].sum().reset_index()
        cohort_pivot = cohort_data.pivot(index='cohort', columns='cohort_index', values='payment_value')
    
    # Calculate retention percentages
    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0) * 100
    retention_matrix = retention_matrix.round(1)
    
    # Calculate key metrics
    metrics = calculate_cohort_metrics_streamlit(retention_matrix, cohort_sizes, retention_type)
    
    # Generate visualizations
    visualizations = create_cohort_visualizations(retention_matrix, cohort_sizes, cohort_period, retention_type)
    
    # Generate insights
    insights = generate_cohort_insights_streamlit(retention_matrix, metrics, retention_type)
    
    return {
        'retention_matrix': retention_matrix,
        'cohort_sizes': cohort_sizes,
        'metrics': metrics,
        'visualizations': visualizations,
        'insights': insights,
        'cohort_period': cohort_period,
        'retention_type': retention_type
    }


def calculate_cohort_metrics_streamlit(retention_matrix, cohort_sizes, retention_type='customers'):
    """
    Calculate comprehensive cohort metrics for Streamlit display.
    """
    metrics = {}
    
    # Basic metrics
    metrics['total_cohorts'] = len(retention_matrix)
    metrics['avg_cohort_size'] = round(cohort_sizes.mean(), 0)
    metrics['total_initial'] = int(cohort_sizes.sum())
    
    # Period-specific retention rates
    periods_to_analyze = min(12, len(retention_matrix.columns))
    
    for period in range(1, min(7, periods_to_analyze)):
        if period in retention_matrix.columns:
            valid_values = retention_matrix[period].dropna()
            if len(valid_values) > 0:
                metrics[f'month_{period}_retention'] = round(valid_values.mean(), 1)
    
    # Overall retention metrics
    if 1 in retention_matrix.columns:
        metrics['month_1_retention_avg'] = round(retention_matrix[1].dropna().mean(), 1)
    
    if 3 in retention_matrix.columns:
        metrics['month_3_retention_avg'] = round(retention_matrix[3].dropna().mean(), 1)
    
    if 6 in retention_matrix.columns:
        metrics['month_6_retention_avg'] = round(retention_matrix[6].dropna().mean(), 1)
    
    if 12 in retention_matrix.columns:
        metrics['month_12_retention_avg'] = round(retention_matrix[12].dropna().mean(), 1)
    
    # Calculate retention curve (average across all cohorts)
    retention_curve = []
    for col in retention_matrix.columns:
        avg_retention = retention_matrix[col].dropna().mean()
        retention_curve.append(avg_retention)
    metrics['retention_curve'] = retention_curve
    
    # Churn rates
    if len(retention_curve) > 1:
        metrics['month_1_churn'] = round(100 - retention_curve[1] if len(retention_curve) > 1 else 0, 1)
        metrics['avg_monthly_churn'] = round(
            np.mean([retention_curve[i-1] - retention_curve[i] 
                    for i in range(1, min(4, len(retention_curve)))]), 1
        )
    
    # Cohort quality score (based on early retention)
    early_retention = retention_matrix[1].dropna() if 1 in retention_matrix.columns else pd.Series([0])
    metrics['cohort_quality_score'] = calculate_quality_score(early_retention)
    
    # Best and worst performing cohorts
    if 1 in retention_matrix.columns:
        best_cohort_retention = retention_matrix[1].max()
        worst_cohort_retention = retention_matrix[1].min()
        metrics['best_cohort'] = {
            'period': str(retention_matrix[retention_matrix[1] == best_cohort_retention].index[0]),
            'retention': round(best_cohort_retention, 1)
        }
        metrics['worst_cohort'] = {
            'period': str(retention_matrix[retention_matrix[1] == worst_cohort_retention].index[0]),
            'retention': round(worst_cohort_retention, 1)
        }
    
    # Retention stability (standard deviation)
    if 1 in retention_matrix.columns:
        metrics['retention_stability'] = round(retention_matrix[1].std(), 1)
    
    return metrics


def calculate_quality_score(retention_series):
    """
    Calculate a quality score for cohorts based on retention patterns.
    """
    if len(retention_series) == 0:
        return 0
    
    avg_retention = retention_series.mean()
    
    if avg_retention >= 40:
        return 'Excellent'
    elif avg_retention >= 30:
        return 'Good'
    elif avg_retention >= 20:
        return 'Fair'
    else:
        return 'Needs Improvement'


def create_cohort_visualizations(retention_matrix, cohort_sizes, cohort_period='month', retention_type='customers'):
    """
    Create multiple visualizations for cohort analysis.
    """
    visualizations = {}
    
    # 1. Retention Heatmap
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(14, 8))
    
    # Prepare data for heatmap
    heatmap_data = retention_matrix.fillna(0)
    
    # Create custom colormap
    if retention_type == 'customers':
        cmap = sns.diverging_palette(10, 150, as_cmap=True)
        fmt = '.0f'
    else:
        cmap = sns.diverging_palette(10, 250, as_cmap=True)
        fmt = '.0f'
    
    # Draw heatmap
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt=fmt,
                cmap=cmap,
                cbar_kws={'label': f'Retention Rate (%)'},
                ax=ax_heatmap,
                vmin=0,
                vmax=100,
                linewidths=0.5,
                linecolor='gray')
    
    ax_heatmap.set_title(f'{retention_type.capitalize()} Retention Cohort Analysis - {cohort_period.capitalize()}ly Cohorts', 
                         fontsize=14, fontweight='bold')
    ax_heatmap.set_xlabel(f'Periods Since First Purchase ({cohort_period}s)', fontsize=12)
    ax_heatmap.set_ylabel(f'Cohort ({cohort_period.capitalize()})', fontsize=12)
    
    # Format y-axis labels (cohort periods)
    ax_heatmap.set_yticklabels([str(period) for period in retention_matrix.index], rotation=0)
    
    plt.tight_layout()
    visualizations['heatmap'] = fig_heatmap
    
    # 2. Retention Curves by Cohort
    fig_curves, ax_curves = plt.subplots(figsize=(14, 8))
    
    # Plot retention curves for each cohort
    for idx, cohort in enumerate(retention_matrix.index[:10]):  # Limit to 10 cohorts for clarity
        values = retention_matrix.loc[cohort].dropna().values
        periods = range(len(values))
        ax_curves.plot(periods, values, marker='o', label=str(cohort), alpha=0.7)
    
    ax_curves.set_title(f'Retention Curves by Cohort', fontsize=14, fontweight='bold')
    ax_curves.set_xlabel(f'Periods Since First Purchase', fontsize=12)
    ax_curves.set_ylabel(f'Retention Rate (%)', fontsize=12)
    ax_curves.grid(True, alpha=0.3)
    ax_curves.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Cohort')
    
    plt.tight_layout()
    visualizations['curves'] = fig_curves
    
    # 3. Average Retention Curve
    fig_avg, ax_avg = plt.subplots(figsize=(12, 6))
    
    # Calculate average retention across all cohorts
    avg_retention = []
    periods = []
    for col in retention_matrix.columns:
        avg = retention_matrix[col].dropna().mean()
        avg_retention.append(avg)
        periods.append(col)
    
    # Plot average retention with confidence interval
    ax_avg.plot(periods, avg_retention, marker='o', linewidth=2, markersize=8, color='#3498db', label='Average Retention')
    
    # Add trend line
    if len(periods) > 1:
        z = np.polyfit(periods, avg_retention, 2)
        p = np.poly1d(z)
        ax_avg.plot(periods, p(periods), "--", alpha=0.5, color='red', label='Trend')
    
    # Highlight key periods
    for i, (period, retention) in enumerate(zip(periods[:7], avg_retention[:7])):
        ax_avg.annotate(f'{retention:.1f}%', 
                       xy=(period, retention), 
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       fontsize=9,
                       color='black',
                       fontweight='bold')
    
    ax_avg.set_title('Average Retention Rate Over Time', fontsize=14, fontweight='bold')
    ax_avg.set_xlabel(f'Periods Since First Purchase', fontsize=12)
    ax_avg.set_ylabel('Average Retention Rate (%)', fontsize=12)
    ax_avg.grid(True, alpha=0.3)
    ax_avg.legend()
    ax_avg.set_ylim(0, 105)
    
    plt.tight_layout()
    visualizations['average_curve'] = fig_avg
    
    # 4. Cohort Size Distribution
    fig_sizes, (ax_sizes1, ax_sizes2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart of cohort sizes
    cohort_labels = [str(c) for c in cohort_sizes.index]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(cohort_sizes)))
    
    bars = ax_sizes1.bar(range(len(cohort_sizes)), cohort_sizes.values, color=colors)
    ax_sizes1.set_title('Cohort Sizes', fontsize=12, fontweight='bold')
    ax_sizes1.set_xlabel('Cohort', fontsize=10)
    ax_sizes1.set_ylabel(f'Number of {"Customers" if retention_type == "customers" else "Revenue ($)"}', fontsize=10)
    ax_sizes1.set_xticks(range(len(cohort_labels)))
    ax_sizes1.set_xticklabels(cohort_labels, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, cohort_sizes.values):
        height = bar.get_height()
        ax_sizes1.text(bar.get_x() + bar.get_width()/2., height,
                      f'{int(value):,}' if retention_type == 'customers' else f'${value:,.0f}',
                      ha='center', va='bottom', fontsize=8)
    
    # Growth trend
    ax_sizes2.plot(range(len(cohort_sizes)), cohort_sizes.values, marker='o', color='#2ecc71', linewidth=2)
    ax_sizes2.fill_between(range(len(cohort_sizes)), cohort_sizes.values, alpha=0.3, color='#2ecc71')
    ax_sizes2.set_title('Cohort Size Trend', fontsize=12, fontweight='bold')
    ax_sizes2.set_xlabel('Cohort', fontsize=10)
    ax_sizes2.set_ylabel(f'Size', fontsize=10)
    ax_sizes2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    visualizations['cohort_sizes'] = fig_sizes
    
    return visualizations


def generate_cohort_insights_streamlit(retention_matrix, metrics, retention_type='customers'):
    """
    Generate actionable insights from cohort analysis for Streamlit display.
    """
    insights = {
        'summary': [],
        'strengths': [],
        'concerns': [],
        'recommendations': []
    }
    
    # Summary insights
    insights['summary'].append(
        f"📊 Analyzed {metrics['total_cohorts']} cohorts with {metrics['total_initial']:,} total {retention_type}"
    )
    
    if 'month_1_retention_avg' in metrics:
        retention_1m = metrics['month_1_retention_avg']
        insights['summary'].append(
            f"📈 Average 1-month retention: {retention_1m:.1f}%"
        )
        
        # Evaluate retention performance
        if retention_type == 'customers':
            if retention_1m >= 40:
                insights['strengths'].append("✅ Excellent customer retention in first month")
            elif retention_1m >= 25:
                insights['strengths'].append("👍 Good customer retention in first month")
            else:
                insights['concerns'].append("⚠️ Low first-month retention indicates onboarding issues")
        
    if 'month_3_retention_avg' in metrics:
        insights['summary'].append(
            f"📉 Average 3-month retention: {metrics['month_3_retention_avg']:.1f}%"
        )
    
    # Cohort quality
    quality_score = metrics.get('cohort_quality_score', 'Unknown')
    insights['summary'].append(f"⭐ Overall cohort quality: {quality_score}")
    
    # Stability analysis
    if 'retention_stability' in metrics:
        stability = metrics['retention_stability']
        if stability > 15:
            insights['concerns'].append(
                f"📊 High retention variability ({stability:.1f}%) suggests inconsistent customer experience"
            )
        else:
            insights['strengths'].append(
                f"📊 Consistent retention across cohorts (std: {stability:.1f}%)"
            )
    
    # Best/Worst cohort analysis
    if 'best_cohort' in metrics and 'worst_cohort' in metrics:
        best = metrics['best_cohort']
        worst = metrics['worst_cohort']
        
        insights['summary'].append(
            f"🏆 Best cohort: {best['period']} with {best['retention']:.1f}% retention"
        )
        insights['summary'].append(
            f"📉 Worst cohort: {worst['period']} with {worst['retention']:.1f}% retention"
        )
        
        # Large gap indicates opportunity
        gap = best['retention'] - worst['retention']
        if gap > 20:
            insights['concerns'].append(
                f"🔍 Large retention gap ({gap:.1f}%) between cohorts - investigate success factors"
            )
    
    # Churn analysis
    if 'month_1_churn' in metrics:
        churn = metrics['month_1_churn']
        if churn > 60:
            insights['concerns'].append(
                f"🔴 High first-month churn ({churn:.1f}%) requires immediate attention"
            )
    
    # Recommendations based on analysis
    if 'month_1_retention_avg' in metrics and metrics['month_1_retention_avg'] < 30:
        insights['recommendations'].extend([
            "🎯 Implement onboarding email sequence for new customers",
            "🎁 Create first-purchase incentive program",
            "📧 Develop welcome series with product education"
        ])
    
    if 'retention_stability' in metrics and metrics['retention_stability'] > 15:
        insights['recommendations'].extend([
            "🔄 Standardize customer experience across acquisition channels",
            "📊 Analyze successful cohorts to identify best practices",
            "🎯 Implement consistent retention strategies"
        ])
    
    if 'avg_monthly_churn' in metrics and metrics['avg_monthly_churn'] > 10:
        insights['recommendations'].extend([
            "💡 Develop win-back campaigns for churned customers",
            "🔔 Implement churn prediction and prevention system",
            "🎁 Create loyalty program to increase retention"
        ])
    
    # Trend analysis
    if 'retention_curve' in metrics and len(metrics['retention_curve']) > 3:
        curve = metrics['retention_curve']
        
        # Check if retention stabilizes
        if len(curve) > 6:
            later_variance = np.std(curve[3:6])
            if later_variance < 3:
                insights['strengths'].append(
                    "📈 Retention stabilizes after month 3, indicating strong core customer base"
                )
        
        # Check retention decay rate
        if len(curve) > 1:
            decay_rate = (curve[0] - curve[-1]) / len(curve)
            if decay_rate > 10:
                insights['concerns'].append(
                    f"📉 Steep retention decline ({decay_rate:.1f}% per period) needs intervention"
                )
    
    return insights


def display_cohort_analysis_streamlit(analysis_results):
    """
    Display cohort analysis results in Streamlit with proper formatting.
    
    Usage in Streamlit app:
    ---------------------
    results = cohort_analysis_dashboard(orders, order_payments, customers)
    display_cohort_analysis_streamlit(results)
    """
    if not analysis_results:
        st.warning("No cohort data available for the selected period")
        return
    
    # Extract components
    metrics = analysis_results['metrics']
    visualizations = analysis_results['visualizations']
    insights = analysis_results['insights']
    retention_matrix = analysis_results['retention_matrix']
    
    # Display header metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Cohorts",
            f"{metrics['total_cohorts']}",
            help="Number of unique cohorts analyzed"
        )
    
    with col2:
        if 'month_1_retention_avg' in metrics:
            st.metric(
                "Avg 1-Month Retention",
                f"{metrics['month_1_retention_avg']:.1f}%",
                delta=f"{metrics['month_1_retention_avg'] - 100:.1f}%"
            )
        else:
            st.metric("Avg 1-Month Retention", "N/A")
    
    with col3:
        if 'month_3_retention_avg' in metrics:
            st.metric(
                "Avg 3-Month Retention",
                f"{metrics['month_3_retention_avg']:.1f}%",
                help="Average retention after 3 months"
            )
        else:
            st.metric("Avg 3-Month Retention", "N/A")
    
    with col4:
        st.metric(
            "Cohort Quality",
            metrics.get('cohort_quality_score', 'N/A'),
            help="Overall quality rating of cohorts"
        )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Retention Matrix", "📈 Trends", "💡 Insights", "📋 Raw Data"])
    
    with tab1:
        st.subheader("Retention Heatmap")
        st.pyplot(visualizations['heatmap'])
        
        st.subheader("Cohort Sizes")
        st.pyplot(visualizations['cohort_sizes'])
    
    with tab2:
        st.subheader("Average Retention Curve")
        st.pyplot(visualizations['average_curve'])
        
        st.subheader("Individual Cohort Retention Curves")
        st.pyplot(visualizations['curves'])
    
    with tab3:
        # Display insights
        st.subheader("📊 Analysis Summary")
        for item in insights['summary']:
            st.write(item)
        
        if insights['strengths']:
            st.subheader("💪 Strengths")
            for strength in insights['strengths']:
                st.success(strength)
        
        if insights['concerns']:
            st.subheader("⚠️ Areas of Concern")
            for concern in insights['concerns']:
                st.warning(concern)
        
        if insights['recommendations']:
            st.subheader("🎯 Recommendations")
            for rec in insights['recommendations']:
                st.info(rec)
    
    with tab4:
        st.subheader("Retention Matrix Data")
        
        # Display retention matrix with formatting
        formatted_matrix = retention_matrix.round(1).fillna('-')
        st.dataframe(
            formatted_matrix.style.background_gradient(cmap='RdYlGn', vmin=0, vmax=100),
            use_container_width=True
        )
        
        # Download button for CSV
        csv = retention_matrix.to_csv()
        st.download_button(
            label="Download Retention Matrix as CSV",
            data=csv,
            file_name=f'cohort_retention_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv'
        )
        
        # Additional metrics table
        st.subheader("Key Metrics")
        metrics_df = pd.DataFrame([
            {'Metric': 'Total Cohorts', 'Value': metrics['total_cohorts']},
            {'Metric': 'Average Cohort Size', 'Value': f"{metrics['avg_cohort_size']:,.0f}"},
            {'Metric': '1-Month Retention', 'Value': f"{metrics.get('month_1_retention_avg', 0):.1f}%"},
            {'Metric': '3-Month Retention', 'Value': f"{metrics.get('month_3_retention_avg', 0):.1f}%"},
            {'Metric': '6-Month Retention', 'Value': f"{metrics.get('month_6_retention_avg', 0):.1f}%"},
            {'Metric': 'Retention Stability (Std Dev)', 'Value': f"{metrics.get('retention_stability', 0):.1f}%"},
        ])
        st.table(metrics_df)


# Example usage in Streamlit app
def example_streamlit_implementation():
    """
    Example of how to integrate cohort analysis in your Streamlit dashboard.
    """
    st.header("🔄 Cohort Analysis")
    
    # Configuration options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cohort_period = st.selectbox(
            "Cohort Period",
            options=['month', 'week', 'quarter'],
            index=0,
            help="How to group cohorts"
        )
    
    with col2:
        retention_type = st.selectbox(
            "Retention Type",
            options=['customers', 'revenue'],
            index=0,
            help="Measure customer count or revenue retention"
        )
    
    with col3:
        max_periods = st.number_input(
            "Max Periods to Analyze",
            min_value=3,
            max_value=24,
            value=12,
            help="Number of periods to track after first purchase"
        )
    
    # Run analysis button
    if st.button("🚀 Run Cohort Analysis", type="primary"):
        with st.spinner("Analyzing cohorts..."):
            # Assume these dataframes are already loaded
            # orders, order_payments, customers = load_data()
            
            # Run cohort analysis
            results = cohort_analysis_dashboard(
                orders_df=orders,
                order_payments_df=order_payments,
                customers_df=customers,
                date_range=date_range,  # From your filters
                cohort_period=cohort_period,
                retention_type=retention_type,
                max_periods=max_periods
            )
            
            # Display results
            if results:
                display_cohort_analysis_streamlit(results)
            else:
                st.error("No data available for cohort analysis")

if __name__ == "__main__":
    print("Cohort Analysis Module loaded successfully")
    print("Use cohort_analysis_dashboard() for analysis")
    print("Use display_cohort_analysis_streamlit() for visualization")



