"""
RFM Customer Segmentation Visualization Module for Streamlit
Designed to work with existing rfm_customer_segmentation function from key_metrics.py
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


def display_rfm_analysis_tab(orders_df, order_payments_df, customers_df=None, 
                             date_range=None, customer_state=None, 
                             payment_method=None, product_category=None):
    """
    Complete RFM analysis visualization for Streamlit dashboard.
    Designed to be added as Tab 4 in the existing dashboard.
    
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
    
    # Import the RFM function from key_metrics
    try:
        from key_metrics import rfm_customer_segmentation, segment_recommendation_engine
    except ImportError:
        st.error("⚠️ Unable to import RFM functions from key_metrics.py")
        st.info("Please ensure key_metrics.py is in your project directory")
        return
    
    # Header and description
    st.subheader("👥 RFM Customer Segmentation Analysis")
    
    with st.expander("ℹ️ What is RFM Analysis?"):
        st.write("""
        **RFM (Recency, Frequency, Monetary)** analysis segments customers based on:
        - **Recency (R)**: How recently did the customer purchase?
        - **Frequency (F)**: How often do they purchase?
        - **Monetary (M)**: How much do they spend?
        
        This helps identify:
        - 🏆 **Champions**: Your best customers
        - 💎 **Loyal Customers**: Frequent buyers
        - 🌟 **Potential Loyalists**: Recent customers with potential
        - 🆕 **New Customers**: First-time buyers
        - 💤 **Hibernating**: Haven't purchased recently
        - ⚠️ **At Risk**: Previously good customers showing decline
        """)
    
    # Configuration options
    col_config1, col_config2, col_config3 = st.columns(3)
    
    with col_config1:
        reference_date = st.date_input(
            "📅 Reference Date for Recency",
            value=orders_df['order_purchase_timestamp'].max(),
            max_value=orders_df['order_purchase_timestamp'].max(),
            help="Date to calculate recency from (typically today or last order date)"
        )
    
    with col_config2:
        show_recommendations = st.checkbox(
            "💡 Show Recommendations",
            value=True,
            help="Display marketing recommendations for each segment"
        )
    
    with col_config3:
        export_data = st.checkbox(
            "📥 Enable Export",
            value=False,
            help="Allow downloading RFM data as CSV"
        )
    
    # Run RFM Analysis button
    if st.button("🔄 Generate RFM Segmentation", type="primary", use_container_width=True):
        
        with st.spinner("Analyzing customer segments..."):
            try:
                # Call the existing RFM function from key_metrics
                rfm_df, segment_summary, recommendations = rfm_customer_segmentation(
                    orders_df=orders_df,
                    order_payments_df=order_payments_df,
                    customers_df=customers_df,
                    date_range=date_range,
                    reference_date=reference_date
                )
                
                # Store results in session state
                st.session_state['rfm_results'] = {
                    'rfm_df': rfm_df,
                    'segment_summary': segment_summary,
                    'recommendations': recommendations,
                    'timestamp': datetime.now()
                }
                
            except Exception as e:
                st.error(f"Error running RFM analysis: {str(e)}")
                st.info("Please check that your data is properly filtered and formatted")
                return
    
    # Display results if available
    if 'rfm_results' in st.session_state:
        results = st.session_state['rfm_results']
        rfm_df = results['rfm_df']
        segment_summary = results['segment_summary']
        recommendations = results['recommendations']
        
        # Display key metrics
        st.markdown("---")
        display_rfm_metrics(rfm_df, segment_summary)
        
        # Create tabs for different visualizations
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "📊 Segment Distribution",
            "📈 RFM Scatter Plot", 
            "🗺️ Segment Matrix",
            "📋 Detailed Data"
        ])
        
        with viz_tab1:
            display_segment_distribution(rfm_df, segment_summary)
        
        with viz_tab2:
            display_rfm_scatter(rfm_df)
        
        with viz_tab3:
            display_rfm_heatmap(rfm_df, segment_summary)
        
        with viz_tab4:
            display_rfm_details(rfm_df, segment_summary, recommendations, export_data)
        
        # Recommendations section
        if show_recommendations and len(recommendations.keys()) != 0:
            st.markdown("---")
            display_segment_recommendations(recommendations)
        
        # Applied filters summary
        st.markdown("---")
        display_filter_summary(date_range, customer_state, payment_method, product_category)
        
    else:
        st.info("👆 Click 'Generate RFM Segmentation' to analyze your customer segments")


def display_rfm_metrics(rfm_df, segment_summary):
    """Display key RFM metrics in metric cards"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_recency = rfm_df['recency'].mean()
        st.metric(
            "📅 Avg Recency",
            f"{avg_recency:.0f} days",
            help="Average days since last purchase"
        )
    
    with col2:
        avg_frequency = rfm_df['frequency'].mean()
        st.metric(
            "🔄 Avg Frequency",
            f"{avg_frequency:.1f}",
            help="Average number of purchases per customer"
        )
    
    with col3:
        total_customers = rfm_df['customer_id'].nunique()
        champions_count = len(rfm_df[rfm_df['segment'] == 'Champions'])
        champions_pct = (champions_count / total_customers * 100) if total_customers > 0 else 0
        st.metric(
            "🏆 Champions",
            f"{champions_pct:.1f}%",
            f"{champions_count:,} customers",
            help="Your best customers"
        )


def display_segment_distribution(rfm_df, segment_summary):
    """Display segment distribution visualizations"""
    
    st.subheader("📊 Customer Segment Distribution")
    
    # Prepare data
    segment_counts = rfm_df['segment'].value_counts().reset_index()
    segment_counts.columns = ['segment', 'count']
    segment_counts['percentage'] = (segment_counts['count'] / segment_counts['count'].sum() * 100).round(1)
    
    # Merge with segment summary for additional metrics
    if not segment_summary.empty and 'segment' in segment_summary.columns:
        segment_data = segment_counts.merge(segment_summary, on='segment', how='left')
    else:
        segment_data = segment_counts
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart using matplotlib
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#f39c12', '#e74c3c', '#1abc9c', '#34495e', '#e67e22']
        explode = [0.05 if seg == 'Champions' else 0.02 for seg in segment_counts['segment']]
        
        wedges, texts, autotexts = ax1.pie(
            segment_counts['count'],
            labels=segment_counts['segment'],
            autopct='%1.1f%%',
            colors=colors[:len(segment_counts)],
            explode=explode,
            shadow=True,
            startangle=90
        )
        
        # Enhance text
        for text in texts:
            text.set_fontsize(11)
            text.set_fontweight('bold')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        ax1.set_title("Customer Segments Distribution", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        st.pyplot(fig1)
    
    with col2:
        # Bar chart
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        
        bars = ax2.bar(segment_counts['segment'], segment_counts['count'], 
                       color=colors[:len(segment_counts)])
        
        # Add value labels
        for bar, count, pct in zip(bars, segment_counts['count'], segment_counts['percentage']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count):,}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax2.set_xlabel("Customer Segment", fontsize=12)
        ax2.set_ylabel("Number of Customers", fontsize=12)
        ax2.set_title("Segment Size Comparison", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
    
    # Segment value analysis
    st.subheader("💰 Segment Value Analysis")
    
    # Calculate segment value metrics
    segment_value = rfm_df.groupby('segment').agg({
        'monetary': ['sum', 'mean'],
        'frequency': 'mean',
        'recency': 'mean',
        'customer_id': 'count'
    }).round(2)
    
    segment_value.columns = ['Total_Revenue', 'Avg_Revenue', 'Avg_Frequency', 'Avg_Recency', 'Customer_Count']
    segment_value['Revenue_Share_%'] = (segment_value['Total_Revenue'] / segment_value['Total_Revenue'].sum() * 100).round(1)
    
    # Sort by total revenue
    segment_value = segment_value.sort_values('Total_Revenue', ascending=False)
    
    # Create revenue contribution chart
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    x_pos = range(len(segment_value))
    bars1 = ax3.bar(x_pos, segment_value['Revenue_Share_%'], 
                    color='#3498db', alpha=0.7, label='Revenue Share %')
    
    # Add customer percentage on secondary axis
    ax3_2 = ax3.twinx()
    customer_pct = (segment_value['Customer_Count'] / segment_value['Customer_Count'].sum() * 100)
    bars2 = ax3_2.bar([x + 0.3 for x in x_pos], customer_pct, 
                      width=0.3, color='#e74c3c', alpha=0.7, label='Customer Share %')
    
    ax3.set_xlabel("Customer Segment", fontsize=12)
    ax3.set_ylabel("Revenue Share (%)", fontsize=12, color='#3498db')
    ax3_2.set_ylabel("Customer Share (%)", fontsize=12, color='#e74c3c')
    ax3.set_title("Revenue vs Customer Distribution by Segment", fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(segment_value.index, rotation=45, ha='right')
    
    # Add value labels
    for bar, val in zip(bars1, segment_value['Revenue_Share_%']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#3498db')
    
    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_2.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax3.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)


def display_rfm_scatter(rfm_df):
    """Display RFM 3D scatter plot"""
    
    st.subheader("📈 RFM Score Distribution")
    
    # Create 2D scatter plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Recency vs Frequency
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        
        segment_colors = {
            'Champions': '#2ecc71',
            'Loyal Customers': '#3498db',
            'Potential Loyalists': '#9b59b6',
            'New Customers': '#f39c12',
            'Promising': '#1abc9c',
            'Need Attention': '#e67e22',
            'About to Sleep': '#95a5a6',
            'At Risk': '#e74c3c',
            'Cannot Lose Them': '#c0392b',
            'Hibernating': '#34495e',
            'Lost': '#2c3e50'
        }
        
        for segment in rfm_df['segment'].unique():
            segment_data = rfm_df[rfm_df['segment'] == segment]
            ax1.scatter(segment_data['recency'], segment_data['frequency'],
                       label=segment, alpha=0.6, s=segment_data['monetary']/10,
                       color=segment_colors.get(segment, '#95a5a6'))
        
        ax1.set_xlabel("Recency (days)", fontsize=12)
        ax1.set_ylabel("Frequency (purchases)", fontsize=12)
        ax1.set_title("Recency vs Frequency by Segment", fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)
    
    with col2:
        # Frequency vs Monetary
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        
        for segment in rfm_df['segment'].unique():
            segment_data = rfm_df[rfm_df['segment'] == segment]
            ax2.scatter(segment_data['frequency'], segment_data['monetary'],
                       label=segment, alpha=0.6, s=100,
                       color=segment_colors.get(segment, '#95a5a6'))
        
        ax2.set_xlabel("Frequency (purchases)", fontsize=12)
        ax2.set_ylabel("Monetary Value ($)", fontsize=12)
        ax2.set_title("Frequency vs Monetary by Segment", fontsize=14, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
    
    # RFM Score distribution
    st.subheader("📊 RFM Score Analysis")
    
    fig3, (ax3_1, ax3_2, ax3_3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Recency distribution
    ax3_1.hist(rfm_df['recency'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    ax3_1.axvline(rfm_df['recency'].median(), color='red', linestyle='--', 
                  label=f'Median: {rfm_df["recency"].median():.0f}')
    ax3_1.set_xlabel("Recency (days)")
    ax3_1.set_ylabel("Number of Customers")
    ax3_1.set_title("Recency Distribution")
    ax3_1.legend()
    ax3_1.grid(axis='y', alpha=0.3)
    
    # Frequency distribution
    ax3_2.hist(rfm_df['frequency'], bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax3_2.axvline(rfm_df['frequency'].median(), color='red', linestyle='--',
                  label=f'Median: {rfm_df["frequency"].median():.0f}')
    ax3_2.set_xlabel("Frequency (orders)")
    ax3_2.set_ylabel("Number of Customers")
    ax3_2.set_title("Frequency Distribution")
    ax3_2.legend()
    ax3_2.grid(axis='y', alpha=0.3)
    
    # Monetary distribution (log scale)
    monetary_data = rfm_df['monetary'][rfm_df['monetary'] > 0]
    ax3_3.hist(np.log10(monetary_data + 1), bins=30, color='#f39c12', alpha=0.7, edgecolor='black')
    ax3_3.set_xlabel("Monetary Value (log scale)")
    ax3_3.set_ylabel("Number of Customers")
    ax3_3.set_title("Monetary Distribution (Log Scale)")
    ax3_3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig3)


def display_rfm_heatmap(rfm_df, segment_summary):
    """Display RFM segment heatmap matrix"""
    
    st.subheader("🗺️ RFM Segmentation Matrix")
    
    # Create RFM matrix
    # Group by R and F quartiles to create matrix
    if 'r_quartile' in rfm_df.columns and 'f_quartile' in rfm_df.columns:
        matrix_data = rfm_df.groupby(['r_quartile', 'f_quartile']).agg({
            'customer_id': 'count',
            'monetary': 'mean'
        }).reset_index()
        
        # Create pivot table for heatmap
        pivot_count = matrix_data.pivot(index='r_quartile', 
                                        columns='f_quartile', 
                                        values='customer_id')
        pivot_value = matrix_data.pivot(index='r_quartile', 
                                        columns='f_quartile', 
                                        values='monetary')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Customer count heatmap
        sns.heatmap(pivot_count, annot=True, fmt='.0f', cmap='YlOrRd',
                   cbar_kws={'label': 'Number of Customers'},
                   ax=ax1)
        ax1.set_title("Customer Distribution by RFM Quartiles", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Frequency Quartile (1=Low, 4=High)", fontsize=11)
        ax1.set_ylabel("Recency Quartile (1=Recent, 4=Old)", fontsize=11)
        ax1.invert_yaxis()
        
        # Average value heatmap
        sns.heatmap(pivot_value, annot=True, fmt='.0f', cmap='Greens',
                   cbar_kws={'label': 'Average Monetary Value ($)'},
                   ax=ax2)
        ax2.set_title("Average Customer Value by RFM Quartiles", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Frequency Quartile (1=Low, 4=High)", fontsize=11)
        ax2.set_ylabel("Recency Quartile (1=Recent, 4=Old)", fontsize=11)
        ax2.invert_yaxis()
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Segment transition matrix
    st.subheader("🔄 Segment Characteristics")
    
    # Calculate segment characteristics
    segment_profiles = rfm_df.groupby('segment').agg({
        'recency': ['mean', 'std'],
        'frequency': ['mean', 'std'],
        'monetary': ['mean', 'std', 'sum'],
        'customer_id': 'count'
    }).round(2)
    
    segment_profiles.columns = ['Avg_Recency', 'Std_Recency', 'Avg_Frequency', 'Std_Frequency',
                                'Avg_Monetary', 'Std_Monetary', 'Total_Revenue', 'Customer_Count']
    
    # Create segment profile heatmap
    fig2, ax = plt.subplots(figsize=(12, 8))
    
    # Normalize data for heatmap
    profile_normalized = segment_profiles[['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']].T
    
    # Invert recency for better interpretation (lower is better)
    profile_normalized.loc['Avg_Recency'] = profile_normalized.loc['Avg_Recency'].max() - profile_normalized.loc['Avg_Recency']
    
    # Normalize to 0-100 scale
    for col in profile_normalized.columns:
        max_val = profile_normalized[col].max()
        if max_val > 0:
            profile_normalized[col] = (profile_normalized[col] / max_val * 100)
    
    sns.heatmap(profile_normalized, annot=True, fmt='.0f', cmap='RdYlGn',
               cbar_kws={'label': 'Relative Score (0-100)'},
               ax=ax, vmin=0, vmax=100)
    
    ax.set_title("Segment Profile Comparison (Normalized Scores)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Customer Segment", fontsize=12)
    ax.set_ylabel("RFM Dimension", fontsize=12)
    ax.set_yticklabels(['Recency (inverted)', 'Frequency', 'Monetary Value'], rotation=0)
    
    plt.tight_layout()
    st.pyplot(fig2)


def display_rfm_details(rfm_df, segment_summary, recommendations, export_data):
    """Display detailed RFM data and tables"""
    
    st.subheader("📋 Detailed Segment Data")
    
    # Segment summary table
    if not segment_summary.empty:
        st.write("**Segment Summary Statistics**")
        
        # Format segment summary for display
        summary_display = segment_summary.copy()
        
        # Format numeric columns
        numeric_cols = summary_display.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if 'avg' in col.lower() or 'mean' in col.lower():
                summary_display[col] = summary_display[col].round(2)
            elif 'count' in col.lower():
                summary_display[col] = summary_display[col].astype(int)
        
        st.dataframe(
            summary_display.style.background_gradient(subset=numeric_cols, cmap='YlOrRd'),
            use_container_width=True
        )
    
    # Top customers by segment
    st.write("**Top Customers by Segment**")
    
    segment_filter = st.selectbox(
        "Select Segment to View",
        options=['All'] + list(rfm_df['segment'].unique()),
        index=0
    )
    
    if segment_filter == 'All':
        display_df = rfm_df.head(100)
    else:
        display_df = rfm_df[rfm_df['segment'] == segment_filter].head(50)
    
    # Format display dataframe
    display_cols = ['customer_id', 'segment', 'recency', 'frequency', 'monetary', 'rfm_score']
    if all(col in display_df.columns for col in display_cols):
        display_df = display_df[display_cols].copy()
        display_df['monetary'] = display_df['monetary'].round(2)
        
        st.dataframe(
            display_df.style.format({
                'recency': '{:.0f} days',
                'frequency': '{:.0f} orders',
                'monetary': '${:,.2f}'
            }),
            use_container_width=True
        )
    
    # Export functionality
    if export_data:
        st.write("**Export Options**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export RFM data
            csv_rfm = rfm_df.to_csv(index=False)
            st.download_button(
                label="📥 Download RFM Data (CSV)",
                data=csv_rfm,
                file_name=f'rfm_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
        
        with col2:
            # Export segment summary
            if not segment_summary.empty:
                csv_summary = segment_summary.to_csv(index=False)
                st.download_button(
                    label="📥 Download Segment Summary (CSV)",
                    data=csv_summary,
                    file_name=f'rfm_segments_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv'
                )
        
        with col3:
            # Export recommendations
            if len(recommendations.keys()) != 0:
                csv_rec = recommendations.to_csv(index=False)
                st.download_button(
                    label="📥 Download Recommendations (CSV)",
                    data=csv_rec,
                    file_name=f'rfm_recommendations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv'
                )


def display_segment_recommendations(recommendations):
    """Display marketing recommendations for each segment"""
    
    st.subheader("💡 Segment-Specific Marketing Recommendations")
    
    # Check if segment_recommendation_engine is available
    try:
        from key_metrics import segment_recommendation_engine
        
        # Get RFM data from session state
        if 'rfm_results' in st.session_state:
            rfm_df = st.session_state['rfm_results']['rfm_df']
            segment_summary = st.session_state['rfm_results']['segment_summary']
            
            # Generate recommendations
            recommendations = segment_recommendation_engine(rfm_df, segment_summary)
    except:
        pass
    
    if len(recommendations.keys()) == 0:
        st.info("No recommendations available. Check if segment_recommendation_engine function exists in key_metrics.py")
        return
    
    # Sort by priority
    if 'priority_score' in recommendations.columns:
        recommendations = recommendations.sort_values('priority_score', ascending=False)
    
    # Display recommendations by segment
    for _, rec in recommendations.iterrows():
        segment = rec.get('segment', 'Unknown')
        
        # Choose icon based on segment
        icon_map = {
            'Champions': '🏆',
            'Loyal Customers': '💎',
            'Potential Loyalists': '🌟',
            'New Customers': '🆕',
            'Promising': '📈',
            'Need Attention': '⚠️',
            'About to Sleep': '😴',
            'At Risk': '🚨',
            'Cannot Lose Them': '🔴',
            'Hibernating': '💤',
            'Lost': '❌'
        }
        
        icon = icon_map.get(segment, '👥')
        
        with st.expander(f"{icon} {segment}", expanded=(segment in ['Champions', 'At Risk'])):
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Segment details
                st.write("**Segment Profile:**")
                if 'customer_count' in rec:
                    st.write(f"- Customers: {int(rec['customer_count']):,}")
                if 'avg_recency' in rec:
                    st.write(f"- Avg Recency: {rec['avg_recency']:.0f} days")
                if 'avg_frequency' in rec:
                    st.write(f"- Avg Frequency: {rec['avg_frequency']:.1f} orders")
                if 'avg_monetary' in rec:
                    st.write(f"- Avg Value: ${rec['avg_monetary']:.2f}")
            
            with col2:
                # Priority and timing
                if 'priority_score' in rec:
                    priority = int(rec['priority_score'])
                    if priority >= 8:
                        st.error(f"Priority: {priority}/10 - HIGH")
                    elif priority >= 5:
                        st.warning(f"Priority: {priority}/10 - MEDIUM")
                    else:
                        st.info(f"Priority: {priority}/10 - LOW")
                
                if 'timing' in rec:
                    st.write(f"**Timing:** {rec['timing']}")
            
            # Marketing recommendations
            st.write("**Recommended Actions:**")
            
            if 'campaign_type' in rec:
                st.write(f"📧 **Campaign Type:** {rec['campaign_type']}")
            
            if 'channel' in rec:
                st.write(f"📡 **Channel:** {rec['channel']}")
            
            # Specific recommendations by segment
            segment_strategies = {
                'Champions': [
                    "🎁 Create VIP loyalty program with exclusive benefits",
                    "📧 Send early access to new products/sales",
                    "📞 Assign dedicated customer success manager",
                    "🎯 Use for referral programs and testimonials"
                ],
                'Loyal Customers': [
                    "💳 Offer loyalty points or cashback programs",
                    "📦 Provide free shipping on all orders",
                    "🎂 Send birthday/anniversary special offers",
                    "📊 Request feedback and product reviews"
                ],
                'Potential Loyalists': [
                    "📧 Implement drip email campaigns with product education",
                    "🎁 Offer second-purchase discount",
                    "👥 Invite to customer community or forum",
                    "📱 Encourage app download with exclusive offers"
                ],
                'New Customers': [
                    "👋 Send welcome series emails (3-5 emails)",
                    "📚 Provide product usage guides and tips",
                    "🎯 Offer first-time buyer discount for next purchase",
                    "📞 Follow-up call/email to ensure satisfaction"
                ],
                'At Risk': [
                    "🚨 Send 'We miss you' campaign immediately",
                    "💰 Offer significant comeback discount (20-30%)",
                    "📊 Survey to understand dissatisfaction reasons",
                    "🎁 Provide surprise gift with next order"
                ],
                'Hibernating': [
                    "🔄 Launch re-engagement campaign with strong offer",
                    "📧 Send product updates and what's new",
                    "🎯 Use retargeting ads on social media",
                    "❓ Conduct win-back survey"
                ],
                'Cannot Lose Them': [
                    "📞 Personal outreach from senior team member",
                    "💎 Offer premium loyalty status",
                    "🎁 Send personalized gift or handwritten note",
                    "📊 Conduct in-depth feedback session"
                ]
            }
            
            strategies = segment_strategies.get(segment, [
                "📧 Send targeted email campaign",
                "🎯 Create personalized offers",
                "📊 Monitor engagement metrics",
                "🔄 Test different messaging approaches"
            ])
            
            for strategy in strategies:
                st.write(f"  • {strategy}")


def display_filter_summary(date_range, customer_state, payment_method, product_category):
    """Display summary of applied filters"""
    
    with st.expander("🔍 Applied Filters", expanded=False):
        filter_text = []
        
        if date_range:
            filter_text.append(f"**Date Range:** {date_range[0]} to {date_range[1]}")
        
        if customer_state:
            filter_text.append(f"**States:** {', '.join(customer_state)}")
        
        if payment_method:
            filter_text.append(f"**Payment Methods:** {', '.join(payment_method)}")
        
        if product_category:
            filter_text.append(f"**Product Categories:** {', '.join(product_category[:3])}..." if len(product_category) > 3 else ', '.join(product_category))
        
        if filter_text:
            for filter_item in filter_text:
                st.write(filter_item)
        else:
            st.write("No filters applied - analyzing all data")


# Integration example for main dashboard
def integrate_rfm_tab_example():
    """
    Example of how to integrate RFM analysis as Tab 4 in your main dashboard
    
    Add this code to your streamlit_app.py file:
    """
    
    code_example = """
    # In your main streamlit_app.py file, modify the tabs section:
    
    # Import the RFM display function
    from rfm_visualization import display_rfm_analysis_tab
    
    # SECTION 6: DETAILED INSIGHTS (Modified to include RFM)
    st.header("📈 Detailed Performance Insights")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Temporal Trends", 
        "🚚 Delivery Performance", 
        "💡 Key Recommendations",
        "👥 RFM Segmentation"  # NEW TAB
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
        # NEW: RFM Analysis Tab
        display_rfm_analysis_tab(
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
    st.info("RFM Visualization Module loaded successfully")
    st.code(integrate_rfm_tab_example(), language='python')
