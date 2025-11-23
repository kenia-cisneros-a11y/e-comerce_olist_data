"""
Customer Lifetime Value (CLV) Visualization Module for Streamlit
Designed to work with existing customer_lifetime_value function from key_metrics.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")


def display_clv_analysis_tab(orders_df, order_payments_df, customers_df=None,
                             date_range=None, customer_state=None, 
                             payment_method=None, product_category=None):
    """
    Complete Customer Lifetime Value analysis visualization for Streamlit dashboard.
    Designed to be added as Tab 5 in the existing dashboard.
    
    Parameters:
    -----------
    orders_df : DataFrame - Filtered orders data
    order_payments_df : DataFrame - Filtered payment data  
    customers_df : DataFrame - Customer data (optional)
    date_range : tuple - Date filter from sidebar
    customer_state : list - State filter from sidebar
    payment_method : list - Payment method filter from sidebar
    product_category : list - Product category filter from sidebar
    """
    
    # Import the CLV function from key_metrics
    try:
        from key_metrics import customer_lifetime_value
    except ImportError:
        st.error("⚠️ Unable to import customer_lifetime_value from key_metrics.py")
        st.info("Please ensure key_metrics.py is in your project directory")
        return
    
    # Header and description
    st.subheader("💰 Customer Lifetime Value Analysis")
    
    with st.expander("ℹ️ What is Customer Lifetime Value (CLV)?"):
        st.write("""
        **Customer Lifetime Value (CLV)** predicts the total revenue a customer will generate over their entire relationship with your business.
        
        **Key Components:**
        - 📅 **Customer Age**: Time since first purchase
        - 🔄 **Purchase Frequency**: How often they buy
        - 💵 **Average Order Value**: Typical spending per order
        - 📈 **Predicted Future Value**: Expected revenue over time
        
        **Why CLV Matters:**
        - 🎯 **Acquisition Cost Justification**: Know how much to spend acquiring customers
        - 💎 **Resource Allocation**: Focus on high-value customers
        - 📊 **Business Valuation**: Total CLV = Business value
        - 🚀 **Growth Strategy**: Identify value drivers
        """)
    
    # Configuration options
    col_config1, col_config2, col_config3, col_config4 = st.columns(4)
    
    with col_config1:
        prediction_months = st.selectbox(
            "📅 Prediction Period",
            options=[3, 6, 12, 24],
            index=1,
            help="Months ahead to predict CLV"
        )
    
    with col_config2:
        show_predictions = st.checkbox(
            "📈 Show Predictions",
            value=True,
            help="Display predicted values and trends"
        )
    
    with col_config3:
        show_cohorts = st.checkbox(
            "👥 Cohort Analysis",
            value=False,
            help="Analyze CLV by customer cohorts"
        )
    
    with col_config4:
        export_data = st.checkbox(
            "📥 Enable Export",
            value=False,
            help="Allow downloading CLV data as CSV"
        )
    
    # Run CLV Analysis button
    if st.button("💎 Calculate Lifetime Values", type="primary", use_container_width=True):
        
        with st.spinner(f"Calculating {prediction_months}-month CLV for all customers..."):
            try:
                # Call the existing CLV function from key_metrics
                clv_df = customer_lifetime_value(
                    orders_df=orders_df,
                    order_payments_df=order_payments_df,
                    months_ahead=prediction_months
                )
                
                # Store results in session state
                st.session_state['clv_results'] = {
                    'clv_df': clv_df,
                    'prediction_months': prediction_months,
                    'timestamp': datetime.now(),
                    'filters': {
                        'date_range': date_range,
                        'states': customer_state,
                        'payment_methods': payment_method,
                        'categories': product_category
                    }
                }
                
            except Exception as e:
                st.error(f"Error calculating CLV: {str(e)}")
                st.info("Please check that your data is properly filtered and formatted")
                return
    
    # Display results if available
    if 'clv_results' in st.session_state:
        results = st.session_state['clv_results']
        clv_df = results['clv_df']
        prediction_months = results['prediction_months']
        
        # Display key metrics
        st.markdown("---")
        display_clv_metrics(clv_df, prediction_months)
        
        # Create tabs for different visualizations
        viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5 = st.tabs([
            "📊 Value Distribution",
            "🎯 Customer Tiers",
            "📈 Predictive Analysis",
            "🔄 Frequency Patterns",
            "📋 Detailed Data"
        ])
        
        with viz_tab1:
            display_clv_distribution(clv_df)
        
        with viz_tab2:
            display_clv_tiers(clv_df)
        
        with viz_tab3:
            if show_predictions:
                display_clv_predictions(clv_df, prediction_months)
            else:
                st.info("Enable 'Show Predictions' to see predictive analysis")
        
        with viz_tab4:
            display_frequency_analysis(clv_df)
        
        with viz_tab5:
            display_clv_details(clv_df, export_data, show_cohorts)
        
        # Customer insights section
        st.markdown("---")
        display_clv_insights(clv_df, prediction_months)
        
        # Applied filters summary
        st.markdown("---")
        display_filter_summary(results['filters'])
        
    else:
        st.info("👆 Click 'Calculate Lifetime Values' to analyze your customer CLV")


def display_clv_metrics(clv_df, prediction_months):
    """Display key CLV metrics in metric cards"""
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        total_customers = len(clv_df)
        st.metric(
            "👥 Total Customers",
            f"{total_customers:,}",
            help="Total unique customers analyzed"
        )
    
    with col2:
        total_current_value = clv_df['total_value'].sum()
        st.metric(
            "💵 Current Value",
            f"${total_current_value:,.0f}",
            help="Total historical revenue from all customers"
        )
    
    with col3:
        total_predicted_clv = clv_df['predicted_clv'].sum()
        st.metric(
            "📈 Predicted CLV",
            f"${total_predicted_clv:,.0f}",
            f"${(total_predicted_clv - total_current_value):,.0f}",
            help=f"Total predicted value over next {prediction_months} months"
        )
    
    with col4:
        avg_clv = clv_df['predicted_clv'].mean()
        st.metric(
            "💰 Avg CLV",
            f"${avg_clv:,.0f}",
            help=f"Average {prediction_months}-month CLV per customer"
        )
    
    with col5:
        avg_frequency = clv_df['purchase_frequency'].mean()
        st.metric(
            "🔄 Avg Frequency",
            f"{avg_frequency:.1f} days",
            help="Average days between purchases"
        )
    
    with col6:
        high_value_pct = (len(clv_df[clv_df['clv_tier'] == 'High Value']) / total_customers * 100) if 'clv_tier' in clv_df.columns else 0
        st.metric(
            "💎 High Value %",
            f"{high_value_pct:.1f}%",
            help="Percentage of high-value customers"
        )


def display_clv_distribution(clv_df):
    """Display CLV distribution visualizations"""
    
    st.subheader("📊 Customer Lifetime Value Distribution")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # CLV Distribution Histogram
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        
        # Log transform for better visualization if values are skewed
        clv_values = clv_df['predicted_clv']
        
        if clv_values.min() > 0:
            # Use log scale if all values are positive
            ax1.hist(np.log10(clv_values + 1), bins=50, color='#3498db', alpha=0.7, edgecolor='black')
            ax1.set_xlabel("Predicted CLV (log scale)", fontsize=12)
            
            # Add percentile lines
            percentiles = [25, 50, 75, 90]
            for p in percentiles:
                val = np.percentile(clv_values, p)
                log_val = np.log10(val + 1)
                ax1.axvline(log_val, color='red', linestyle='--', alpha=0.5)
                ax1.text(log_val, ax1.get_ylim()[1]*0.9, f'{p}%\n${val:,.0f}', 
                        ha='center', fontsize=9)
        else:
            ax1.hist(clv_values, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
            ax1.set_xlabel("Predicted CLV ($)", fontsize=12)
        
        ax1.set_ylabel("Number of Customers", fontsize=12)
        ax1.set_title("CLV Distribution", fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)
        
        # Statistics box
        st.info(f"""
        **Distribution Statistics:**
        - Mean CLV: ${clv_df['predicted_clv'].mean():,.2f}
        - Median CLV: ${clv_df['predicted_clv'].median():,.2f}
        - Std Dev: ${clv_df['predicted_clv'].std():,.2f}
        - Top 10% threshold: ${clv_df['predicted_clv'].quantile(0.9):,.2f}
        """)
    
    with col2:
        # Current vs Predicted Value Scatter
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        
        # Color by CLV tier if available
        if 'clv_tier' in clv_df.columns:
            tier_colors = {
                'High Value': '#2ecc71',
                'Medium Value': '#f39c12',
                'Low Value': '#e74c3c'
            }
            
            for tier in clv_df['clv_tier'].unique():
                tier_data = clv_df[clv_df['clv_tier'] == tier]
                ax2.scatter(tier_data['total_value'], tier_data['predicted_clv'],
                          alpha=0.6, s=50, label=tier,
                          color=tier_colors.get(tier, '#95a5a6'))
        else:
            ax2.scatter(clv_df['total_value'], clv_df['predicted_clv'],
                      alpha=0.6, s=50, color='#3498db')
        
        # Add diagonal reference line
        max_val = max(clv_df['total_value'].max(), clv_df['predicted_clv'].max())
        ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='1:1 Line')
        
        ax2.set_xlabel("Current Value ($)", fontsize=12)
        ax2.set_ylabel("Predicted CLV ($)", fontsize=12)
        ax2.set_title("Current vs Predicted Customer Value", fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        
        # Growth potential box
        growth_potential = ((clv_df['predicted_clv'] - clv_df['total_value']) / clv_df['total_value'] * 100).mean()
        st.success(f"""
        **Growth Potential:**
        - Average growth: {growth_potential:.1f}%
        - Total opportunity: ${(clv_df['predicted_clv'] - clv_df['total_value']).sum():,.0f}
        - Customers with >100% growth: {len(clv_df[clv_df['predicted_clv'] > clv_df['total_value'] * 2]):,}
        """)


def display_clv_tiers(clv_df):
    """Display CLV tier analysis"""
    
    st.subheader("🎯 Customer Value Tiers")
    
    # Create tiers if not present
    if 'clv_tier' not in clv_df.columns:
        clv_df['clv_tier'] = pd.qcut(clv_df['predicted_clv'], 
                                     q=[0, 0.33, 0.67, 1], 
                                     labels=['Low Value', 'Medium Value', 'High Value'])
    
    # Calculate tier statistics
    tier_stats = clv_df.groupby('clv_tier').agg({
        'customer_id': 'count',
        'predicted_clv': ['mean', 'sum'],
        'total_value': 'mean',
        'purchase_count': 'mean',
        'avg_order_value': 'mean',
        'purchase_frequency': 'mean'
    }).round(2)
    
    tier_stats.columns = ['Customer_Count', 'Avg_CLV', 'Total_CLV', 
                          'Avg_Current_Value', 'Avg_Purchases', 
                          'Avg_Order_Value', 'Avg_Frequency']
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue contribution by tier
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        
        revenue_pct = (tier_stats['Total_CLV'] / tier_stats['Total_CLV'].sum() * 100)
        customer_pct = (tier_stats['Customer_Count'] / tier_stats['Customer_Count'].sum() * 100)
        
        x = np.arange(len(tier_stats.index))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, revenue_pct, width, label='Revenue Share %', color='#3498db')
        bars2 = ax2.bar(x + width/2, customer_pct, width, label='Customer Share %', color='#e74c3c')
        
        ax2.set_xlabel("Customer Value Tier", fontsize=12)
        ax2.set_ylabel("Percentage (%)", fontsize=12)
        ax2.set_title("Revenue vs Customer Distribution", fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(tier_stats.index)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig2)
    
    with col2:
        pass
    
    # Tier comparison table
    st.subheader("📊 Tier Characteristics Comparison")
    
    # Format the statistics for display
    display_stats = tier_stats.copy()
    display_stats['Avg_CLV'] = display_stats['Avg_CLV'].apply(lambda x: f'${x:,.0f}')
    display_stats['Total_CLV'] = display_stats['Total_CLV'].apply(lambda x: f'${x:,.0f}')
    display_stats['Avg_Current_Value'] = display_stats['Avg_Current_Value'].apply(lambda x: f'${x:,.0f}')
    display_stats['Avg_Order_Value'] = display_stats['Avg_Order_Value'].apply(lambda x: f'${x:.2f}')
    display_stats['Avg_Frequency'] = display_stats['Avg_Frequency'].apply(lambda x: f'{x:.1f} days')
    
    st.dataframe(display_stats, use_container_width=True)
    
    # Value concentration analysis
    st.subheader("💰 Value Concentration")
    
    # Calculate cumulative value
    clv_sorted = clv_df.sort_values('predicted_clv', ascending=False).copy()
    clv_sorted['cumulative_pct'] = (clv_sorted['predicted_clv'].cumsum() / clv_sorted['predicted_clv'].sum() * 100)
    clv_sorted['customer_pct'] = (np.arange(1, len(clv_sorted) + 1) / len(clv_sorted) * 100)
    
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    ax3.plot(clv_sorted['customer_pct'], clv_sorted['cumulative_pct'], 
            linewidth=2, color='#3498db')
    ax3.fill_between(clv_sorted['customer_pct'], clv_sorted['cumulative_pct'], 
                     alpha=0.3, color='#3498db')
    
    # Add reference lines
    ax3.plot([0, 100], [0, 100], 'r--', alpha=0.5, label='Perfect equality')
    
    # Add 80/20 rule markers
    twenty_pct_idx = int(len(clv_sorted) * 0.2)
    twenty_pct_value = clv_sorted.iloc[twenty_pct_idx]['cumulative_pct']
    ax3.axvline(20, color='green', linestyle=':', alpha=0.7)
    ax3.axhline(twenty_pct_value, color='green', linestyle=':', alpha=0.7)
    ax3.text(25, twenty_pct_value - 5, f'Top 20% = {twenty_pct_value:.1f}% of value', 
            fontsize=10, fontweight='bold')
    
    ax3.set_xlabel("Cumulative % of Customers", fontsize=12)
    ax3.set_ylabel("Cumulative % of CLV", fontsize=12)
    ax3.set_title("Customer Value Concentration (Lorenz Curve)", fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    plt.tight_layout()
    st.pyplot(fig3)


def display_clv_predictions(clv_df, prediction_months):
    """Display CLV predictive analysis"""
    
    st.subheader("📈 Predictive Value Analysis")
    
    # Calculate growth metrics
    clv_df['growth_potential'] = clv_df['predicted_clv'] - clv_df['total_value']
    clv_df['growth_rate'] = (clv_df['growth_potential'] / clv_df['total_value'] * 100).fillna(0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Growth potential distribution
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        
        growth_values = clv_df['growth_potential']
        positive_growth = growth_values[growth_values > 0]
        negative_growth = growth_values[growth_values <= 0]
        
        ax1.hist([positive_growth, negative_growth], bins=30, 
                label=['Positive Growth', 'Negative/No Growth'],
                color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
        
        ax1.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.5)
        ax1.axvline(growth_values.mean(), color='blue', linestyle='--', linewidth=2, 
                   label=f'Mean: ${growth_values.mean():,.0f}')
        
        ax1.set_xlabel(f"Growth Potential over {prediction_months} months ($)", fontsize=12)
        ax1.set_ylabel("Number of Customers", fontsize=12)
        ax1.set_title("Customer Growth Potential Distribution", fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)
        
        # Growth statistics
        st.info(f"""
        **Growth Potential Analysis:**
        - Customers with growth: {len(positive_growth):,} ({len(positive_growth)/len(clv_df)*100:.1f}%)
        - Average growth: ${growth_values.mean():,.2f}
        - Total opportunity: ${growth_values.sum():,.0f}
        - Top 10% growth: ${growth_values.quantile(0.9):,.2f}
        """)
    
    with col2:
        try:
            # Purchase frequency vs CLV
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            
            # Remove outliers for better visualization
            freq_q99 = clv_df['purchase_frequency'].quantile(0.99)
            clv_q99 = clv_df['predicted_clv'].quantile(0.99)
            
            plot_data = clv_df[(clv_df['purchase_frequency'] <= freq_q99) & 
                            (clv_df['predicted_clv'] <= clv_q99)]
            
            scatter = ax2.scatter(plot_data['purchase_frequency'], 
                                plot_data['predicted_clv'],
                                c=plot_data['purchase_count'], 
                                cmap='viridis', 
                                alpha=0.6, 
                                s=50)
            
            # Add trend line
            z = np.polyfit(plot_data['purchase_frequency'], plot_data['predicted_clv'], 1)
            p = np.poly1d(z)
            ax2.plot(plot_data['purchase_frequency'], 
                    p(plot_data['purchase_frequency']), 
                    "r--", alpha=0.5, label=f'Trend')
            
            plt.colorbar(scatter, ax=ax2, label='Purchase Count')
            ax2.set_xlabel("Purchase Frequency (days between purchases)", fontsize=12)
            ax2.set_ylabel(f"Predicted {prediction_months}-Month CLV ($)", fontsize=12)
            ax2.set_title("Frequency vs Lifetime Value", fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)
            
            # Frequency insights
            high_freq = clv_df[clv_df['purchase_frequency'] <= 30]
            st.success(f"""
            **Frequency Insights:**
            - High-frequency customers (< 30 days): {len(high_freq):,}
            - Their avg CLV: ${high_freq['predicted_clv'].mean():,.2f}
            - CLV difference vs others: +{(high_freq['predicted_clv'].mean() / clv_df['predicted_clv'].mean() - 1) * 100:.1f}%
            """)
        except Exception as e:
            st.error(f"Error generating frequency vs CLV plot: {str(e)}")
    
    # Cohort-based predictions if customer age is available
    if 'customer_age_months' in clv_df.columns:
        st.subheader("📅 CLV by Customer Age")
        
        fig3, (ax3_1, ax3_2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # CLV by customer age
        age_clv = clv_df.groupby(pd.cut(clv_df['customer_age_months'], 
                                        bins=[0, 3, 6, 12, 24, 100],
                                        labels=['0-3m', '3-6m', '6-12m', '12-24m', '24m+']))['predicted_clv'].agg(['mean', 'count'])
        
        ax3_1.bar(age_clv.index, age_clv['mean'], color='#3498db', alpha=0.7)
        ax3_1.set_xlabel("Customer Age", fontsize=12)
        ax3_1.set_ylabel("Average CLV ($)", fontsize=12)
        ax3_1.set_title("CLV by Customer Age Group", fontsize=14, fontweight='bold')
        ax3_1.grid(axis='y', alpha=0.3)
        
        # Add count labels
        for i, (val, count) in enumerate(zip(age_clv['mean'], age_clv['count'])):
            ax3_1.text(i, val, f'${val:,.0f}\n(n={count})', ha='center', va='bottom', fontsize=9)
        
        # Purchase patterns by age
        age_patterns = clv_df.groupby(pd.cut(clv_df['customer_age_months'], 
                                             bins=[0, 3, 6, 12, 24, 100],
                                             labels=['0-3m', '3-6m', '6-12m', '12-24m', '24m+'])).agg({
            'purchase_count': 'mean',
            'avg_order_value': 'mean'
        })
        
        ax3_2_twin = ax3_2.twinx()
        
        bars = ax3_2.bar(age_patterns.index, age_patterns['purchase_count'], 
                        color='#2ecc71', alpha=0.7, label='Avg Purchases')
        line = ax3_2_twin.plot(age_patterns.index, age_patterns['avg_order_value'], 
                               'ro-', linewidth=2, markersize=8, label='Avg Order Value')
        
        ax3_2.set_xlabel("Customer Age", fontsize=12)
        ax3_2.set_ylabel("Average Purchase Count", fontsize=12, color='#2ecc71')
        ax3_2_twin.set_ylabel("Average Order Value ($)", fontsize=12, color='red')
        ax3_2.set_title("Purchase Behavior by Customer Age", fontsize=14, fontweight='bold')
        ax3_2.grid(axis='y', alpha=0.3)
        
        # Combine legends
        lines, labels = ax3_2.get_legend_handles_labels()
        lines2, labels2 = ax3_2_twin.get_legend_handles_labels()
        ax3_2.legend(lines + lines2, labels + labels2, loc='upper left')
        
        plt.tight_layout()
        st.pyplot(fig3)


def display_frequency_analysis(clv_df):
    """Display purchase frequency analysis"""
    
    st.subheader("🔄 Purchase Frequency Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Frequency distribution
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        
        # Remove extreme outliers for better visualization
        freq_data = clv_df['purchase_frequency'][clv_df['purchase_frequency'] <= clv_df['purchase_frequency'].quantile(0.95)]
        
        ax1.hist(freq_data, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax1.axvline(freq_data.median(), color='red', linestyle='--', linewidth=2, 
                   label=f'Median: {freq_data.median():.1f} days')
        ax1.axvline(freq_data.mean(), color='blue', linestyle='--', linewidth=2, 
                   label=f'Mean: {freq_data.mean():.1f} days')
        
        ax1.set_xlabel("Days Between Purchases", fontsize=12)
        ax1.set_ylabel("Number of Customers", fontsize=12)
        ax1.set_title("Purchase Frequency Distribution", fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)
        
        # Frequency segments
        freq_segments = pd.cut(clv_df['purchase_frequency'], 
                              bins=[0, 7, 30, 90, 365, float('inf')],
                              labels=['Weekly', 'Monthly', 'Quarterly', 'Yearly', 'Rare'])
        
        freq_dist = freq_segments.value_counts()
        
        st.info(f"""
        **Frequency Segments:**
        - Weekly (< 7 days): {freq_dist.get('Weekly', 0):,} customers
        - Monthly (7-30 days): {freq_dist.get('Monthly', 0):,} customers
        - Quarterly (30-90 days): {freq_dist.get('Quarterly', 0):,} customers
        - Yearly (90-365 days): {freq_dist.get('Yearly', 0):,} customers
        - Rare (> 365 days): {freq_dist.get('Rare', 0):,} customers
        """)
    
    with col2:
        # Purchase count distribution
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        
        purchase_counts = clv_df['purchase_count'].value_counts().sort_index()
        
        # Limit to first 20 values for clarity
        if len(purchase_counts) > 20:
            purchase_counts = purchase_counts.iloc[:20]
        
        ax2.bar(purchase_counts.index, purchase_counts.values, 
               color='#e67e22', alpha=0.7, edgecolor='black')
        
        ax2.set_xlabel("Number of Purchases", fontsize=12)
        ax2.set_ylabel("Number of Customers", fontsize=12)
        ax2.set_title("Purchase Count Distribution", fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add percentage labels for first few bars
        total = purchase_counts.sum()
        for i, (idx, val) in enumerate(purchase_counts.head(5).items()):
            ax2.text(idx, val, f'{val:,}\n({val/total*100:.1f}%)', 
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig2)
        
        # Repeat purchase rate
        single_purchase = len(clv_df[clv_df['purchase_count'] == 1])
        repeat_purchase = len(clv_df[clv_df['purchase_count'] > 1])
        repeat_rate = repeat_purchase / len(clv_df) * 100
        
        st.success(f"""
        **Purchase Behavior:**
        - Single purchase: {single_purchase:,} ({single_purchase/len(clv_df)*100:.1f}%)
        - Repeat customers: {repeat_purchase:,} ({repeat_rate:.1f}%)
        - Avg purchases per customer: {clv_df['purchase_count'].mean():.2f}
        - Max purchases by a customer: {clv_df['purchase_count'].max()}
        """)
    
    # Average order value analysis
    st.subheader("💵 Order Value Analysis")
    
    fig3, (ax3_1, ax3_2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # AOV distribution
    aov_data = clv_df['avg_order_value'][clv_df['avg_order_value'] <= clv_df['avg_order_value'].quantile(0.95)]
    
    ax3_1.hist(aov_data, bins=30, color='#16a085', alpha=0.7, edgecolor='black')
    ax3_1.axvline(aov_data.median(), color='red', linestyle='--', linewidth=2,
                 label=f'Median: ${aov_data.median():.2f}')
    ax3_1.set_xlabel("Average Order Value ($)", fontsize=12)
    ax3_1.set_ylabel("Number of Customers", fontsize=12)
    ax3_1.set_title("Average Order Value Distribution", fontsize=14, fontweight='bold')
    ax3_1.legend()
    ax3_1.grid(axis='y', alpha=0.3)
    
    # AOV vs Purchase Count
    # Group by purchase count and calculate mean AOV
    aov_by_count = clv_df[clv_df['purchase_count'] <= 10].groupby('purchase_count')['avg_order_value'].mean()
    
    ax3_2.plot(aov_by_count.index, aov_by_count.values, 'bo-', linewidth=2, markersize=8)
    ax3_2.fill_between(aov_by_count.index, aov_by_count.values, alpha=0.3, color='blue')
    ax3_2.set_xlabel("Number of Purchases", fontsize=12)
    ax3_2.set_ylabel("Average Order Value ($)", fontsize=12)
    ax3_2.set_title("Order Value vs Purchase Frequency", fontsize=14, fontweight='bold')
    ax3_2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig3)


def display_clv_details(clv_df, export_data, show_cohorts):
    """Display detailed CLV data and tables"""
    
    st.subheader("📋 Detailed Customer Data")
    
    # Summary statistics
    st.write("**CLV Summary Statistics**")
    
    summary_stats = clv_df[['total_value', 'predicted_clv', 'purchase_count', 
                            'avg_order_value', 'purchase_frequency']].describe()
    
    summary_stats = summary_stats.round(2)
    st.dataframe(summary_stats, use_container_width=True)
    
    # Top customers
    st.write("**Top Customers by CLV**")
    
    # Selection options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top_n = st.selectbox("Number of customers to show", [10, 25, 50, 100], index=0)
    
    with col2:
        sort_by = st.selectbox("Sort by", 
                              ['predicted_clv', 'total_value', 'growth_potential', 'purchase_count'],
                              format_func=lambda x: x.replace('_', ' ').title())
    
    with col3:
        if 'clv_tier' in clv_df.columns:
            tier_filter = st.selectbox("Filter by tier", 
                                       ['All'] + list(clv_df['clv_tier'].unique()))
        else:
            tier_filter = 'All'
    
    # Filter and sort data
    if tier_filter != 'All' and 'clv_tier' in clv_df.columns:
        display_df = clv_df[clv_df['clv_tier'] == tier_filter].copy()
    else:
        display_df = clv_df.copy()
    
    # Calculate growth potential if not present
    if 'growth_potential' not in display_df.columns:
        display_df['growth_potential'] = display_df['predicted_clv'] - display_df['total_value']
    
    # Sort and select top N
    display_df = display_df.nlargest(top_n, sort_by)
    
    # Select columns to display
    display_cols = ['customer_id', 'total_value', 'predicted_clv', 'growth_potential',
                   'purchase_count', 'avg_order_value', 'purchase_frequency']
    
    if 'clv_tier' in display_df.columns:
        display_cols.append('clv_tier')
    
    if 'first_purchase' in display_df.columns:
        display_cols.insert(1, 'first_purchase')
    
    # Format for display
    formatted_df = display_df[display_cols].copy()
    formatted_df['total_value'] = formatted_df['total_value'].apply(lambda x: f'${x:,.2f}')
    formatted_df['predicted_clv'] = formatted_df['predicted_clv'].apply(lambda x: f'${x:,.2f}')
    formatted_df['growth_potential'] = formatted_df['growth_potential'].apply(lambda x: f'${x:,.2f}')
    formatted_df['avg_order_value'] = formatted_df['avg_order_value'].apply(lambda x: f'${x:.2f}')
    formatted_df['purchase_frequency'] = formatted_df['purchase_frequency'].apply(lambda x: f'{x:.1f} days')
    
    st.dataframe(formatted_df, use_container_width=True)
    
    # Cohort analysis if requested
    if show_cohorts and 'first_purchase' in clv_df.columns:
        st.write("**CLV by Acquisition Cohort**")
        
        # Create monthly cohorts
        clv_df['cohort_month'] = pd.to_datetime(clv_df['first_purchase']).dt.to_period('M')
        
        cohort_analysis = clv_df.groupby('cohort_month').agg({
            'customer_id': 'count',
            'predicted_clv': ['mean', 'sum'],
            'total_value': 'mean',
            'purchase_count': 'mean'
        }).round(2)
        
        cohort_analysis.columns = ['Customer_Count', 'Avg_CLV', 'Total_CLV', 
                                   'Avg_Current_Value', 'Avg_Purchases']
        
        # Sort by cohort month
        cohort_analysis = cohort_analysis.sort_index()
        
        # Format for display
        cohort_display = cohort_analysis.copy()
        cohort_display['Avg_CLV'] = cohort_display['Avg_CLV'].apply(lambda x: f'${x:,.0f}')
        cohort_display['Total_CLV'] = cohort_display['Total_CLV'].apply(lambda x: f'${x:,.0f}')
        cohort_display['Avg_Current_Value'] = cohort_display['Avg_Current_Value'].apply(lambda x: f'${x:,.0f}')
        cohort_display.index = cohort_display.index.astype(str)
        
        st.dataframe(cohort_display, use_container_width=True)
    
    # Export functionality
    if export_data:
        st.write("**Export Options**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export CLV data
            csv_clv = clv_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CLV Data (CSV)",
                data=csv_clv,
                file_name=f'clv_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
        
        with col2:
            # Export summary statistics
            csv_summary = summary_stats.to_csv()
            st.download_button(
                label="📥 Download Summary Stats (CSV)",
                data=csv_summary,
                file_name=f'clv_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
        
        with col3:
            # Export tier analysis if available
            if 'clv_tier' in clv_df.columns:
                tier_export = clv_df.groupby('clv_tier').agg({
                    'customer_id': 'count',
                    'predicted_clv': ['mean', 'sum'],
                    'total_value': 'mean'
                }).round(2)
                csv_tiers = tier_export.to_csv()
                st.download_button(
                    label="📥 Download Tier Analysis (CSV)",
                    data=csv_tiers,
                    file_name=f'clv_tiers_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv'
                )


def display_clv_insights(clv_df, prediction_months):
    """Display strategic CLV insights and recommendations"""
    
    st.subheader("💡 Strategic CLV Insights")
    
    # Calculate key metrics for insights
    total_customers = len(clv_df)
    total_current_value = clv_df['total_value'].sum()
    total_predicted_clv = clv_df['predicted_clv'].sum()
    growth_opportunity = total_predicted_clv - total_current_value
    
    # Customer concentration
    top_20_pct = int(total_customers * 0.2)
    top_20_value = clv_df.nlargest(top_20_pct, 'predicted_clv')['predicted_clv'].sum()
    concentration_ratio = top_20_value / total_predicted_clv * 100
    
    # Repeat purchase rate
    repeat_customers = len(clv_df[clv_df['purchase_count'] > 1])
    repeat_rate = repeat_customers / total_customers * 100
    
    # Create insight columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 Key Findings**")
        
        findings = []
        
        # Concentration insight
        if concentration_ratio > 70:
            findings.append(f"⚠️ **High Value Concentration**: Top 20% customers = {concentration_ratio:.1f}% of CLV")
            findings.append("   → Risk: Over-dependence on few customers")
            findings.append("   → Action: Diversification and retention focus")
        elif concentration_ratio < 40:
            findings.append(f"���️ **Low Value Concentration**: Top 20% customers = {concentration_ratio:.1f}% of CLV")
            findings.append("   → Risk: Potential for lost value")
            findings.append("   → Action: Focus on customer acquisition")
        elif concentration_ratio >= 20 and concentration_ratio < 40:
            findings.append(f"��� **Moderate Value Concentration**: Top 20% customers = {concentration_ratio:.1f}% of CLV")
            findings.append("   → Risk: Potential for lost value")
            findings.append("   → Action: Focus on customer acquisition and retention")
        elif concentration_ratio > 0 and concentration_ratio < 20:
            findings.append(f"⚪ **Very Low Value Concentration**: Top 20% customers = {concentration_ratio:.1f}% of CLV")
            findings.append("   → Risk: Weak customer value base")
            findings.append("   → Action: Strengthen customer relationships")
        else:
            findings.append(f"�� **No Value Concentration**: No top 20% customers")
            findings.append("   → Risk: No significant value for customers")
            findings.append("   → Action: Focus on customer acquisition/retention")

        
        # Repeat purchase insight
        if repeat_rate > 30:
            findings.append(f"✅ **Strong Repeat Business**: {repeat_rate:.1f}% are repeat customers")
        else:
            findings.append(f"⚠️ **Low Repeat Rate**: Only {repeat_rate:.1f}% make repeat purchases")
            findings.append("   → Focus on first-purchase experience")
        
        # Growth opportunity
        growth_pct = (growth_opportunity / total_current_value * 100)
        findings.append(f"📈 **Growth Potential**: ${growth_opportunity:,.0f} ({growth_pct:.1f}% increase)")
        
        # Average CLV vs acquisition cost rule of thumb
        avg_clv = clv_df['predicted_clv'].mean()
        findings.append(f"💰 **Avg {prediction_months}-Month CLV**: ${avg_clv:,.2f}")
        findings.append(f"   → Max viable CAC: ${avg_clv * 0.3:,.2f} (30% rule)")
        
        for finding in findings:
            st.write(finding)
    
    with col2:
        st.write("**🎯 Strategic Recommendations**")
        
        recommendations = []
        
        # Based on CLV tiers
        if 'clv_tier' in clv_df.columns:
            high_value_pct = len(clv_df[clv_df['clv_tier'] == 'High Value']) / total_customers * 100
            
            if high_value_pct < 20:
                recommendations.append("🔴 **Expand High-Value Segment**:")
                recommendations.append("   • Launch premium tier/services")
                recommendations.append("   • Implement upselling programs")
                recommendations.append("   • Create VIP experiences")
            
            low_value_pct = len(clv_df[clv_df['clv_tier'] == 'Low Value']) / total_customers * 100
            if low_value_pct > 50:
                recommendations.append("⚠️ **Optimize Low-Value Segment**:")
                recommendations.append("   • Automate service delivery")
                recommendations.append("   • Implement self-service options")
                recommendations.append("   • Focus on margin improvement")
        
        # Based on frequency
        avg_frequency = clv_df['purchase_frequency'].mean()
        if avg_frequency > 60:
            recommendations.append("📅 **Increase Purchase Frequency**:")
            recommendations.append("   • Launch subscription programs")
            recommendations.append("   • Create replenishment reminders")
            recommendations.append("   • Develop habit-forming features")
        
        # Based on growth potential
        high_growth = clv_df[clv_df['predicted_clv'] > clv_df['total_value'] * 2]
        if len(high_growth) > total_customers * 0.1:
            recommendations.append(f"🚀 **{len(high_growth):,} High-Growth Customers**:")
            recommendations.append("   • Prioritize engagement")
            recommendations.append("   • Personalized campaigns")
            recommendations.append("   • Exclusive offers")
        
        for rec in recommendations:
            st.write(rec)
    
    # Action matrix
    st.write("**🎬 CLV-Based Action Matrix**")
    
    action_matrix = pd.DataFrame({
        'Customer Segment': ['High CLV + High Frequency', 'High CLV + Low Frequency', 
                            'Low CLV + High Frequency', 'Low CLV + Low Frequency'],
        'Strategy': ['Retain & Reward', 'Increase Frequency', 'Increase Value', 'Evaluate & Optimize'],
        'Tactics': ['VIP programs, Early access, Personal service',
                   'Subscription offers, Reminders, Bundling',
                   'Upselling, Cross-selling, Premium features',
                   'Automation, Self-service, Cost reduction'],
        'Expected Impact': ['Maintain 90%+ retention', '+30% purchase frequency',
                          '+50% order value', '20% cost reduction']
    })
    
    st.table(action_matrix)


def display_filter_summary(filters):
    """Display summary of applied filters"""
    
    with st.expander("🔍 Applied Filters", expanded=False):
        filter_text = []
        
        if filters.get('date_range'):
            filter_text.append(f"**Date Range:** {filters['date_range'][0]} to {filters['date_range'][1]}")
        
        if filters.get('states'):
            filter_text.append(f"**States:** {', '.join(filters['states'])}")
        
        if filters.get('payment_methods'):
            filter_text.append(f"**Payment Methods:** {', '.join(filters['payment_methods'])}")
        
        if filters.get('categories'):
            cats = filters['categories']
            filter_text.append(f"**Product Categories:** {', '.join(cats[:3])}..." if len(cats) > 3 else ', '.join(cats))
        
        if filter_text:
            for filter_item in filter_text:
                st.write(filter_item)
        else:
            st.write("No filters applied - analyzing all data")
        
        st.write(f"**Analysis Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# Integration example for main dashboard
def integrate_clv_tab_example():
    """
    Example of how to integrate CLV analysis as Tab 5 in your main dashboard
    """
    
    code_example = """
    # In your main streamlit_app.py file, modify the tabs section:
    
    # Import the CLV display function
    from clv_visualization import display_clv_analysis_tab
    
    # SECTION 6: DETAILED INSIGHTS (Modified to include CLV)
    st.header("📈 Detailed Performance Insights")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Temporal Trends", 
        "🚚 Delivery Performance", 
        "💡 Key Recommendations",
        "👥 RFM Segmentation",
        "💰 Lifetime Value"  # NEW TAB
    ])
    
    with tab1:
        # Your existing temporal trends code
        pass
    
    with tab2:
        # Your existing delivery performance code
        pass
    
    with tab3:
        # Your existing recommendations code
        pass
    
    with tab4:
        # Your existing RFM code
        pass
    
    with tab5:
        # NEW: CLV Analysis Tab
        display_clv_analysis_tab(
            orders_df=filtered_orders,
            order_payments_df=filtered_payments,
            customers_df=customers,
            date_range=date_range,
            customer_state=customer_state,
            payment_method=payment_method,
            product_category=product_category
        )
    """
    
    return code_example


if __name__ == "__main__":
    st.info("CLV Visualization Module loaded successfully")
    st.code(integrate_clv_tab_example(), language='python')