import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import sys
import os
import warnings
from datetime import datetime, timedelta

# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from utils import *
from key_metrics import *
from cohort_analysis import (
    cohort_analysis_dashboard, 
    display_cohort_analysis_streamlit
)
from rfm_visualization import display_rfm_analysis_tab
from customer_lifetime_value_visualization import display_clv_analysis_tab
from streamlit_prophet_integration import add_prophet_tab

# Page config
st.set_page_config(
    page_title="Executive E-commerce Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    orders_df = load_table('orders')
    order_payments_df = load_table('order_payments')
    customers_df = load_table('customers')
    products_df = load_table('products')
    order_items_df = load_table('order_items')
    order_reviews_df = load_table('order_reviews')
    product_category_name_translation_df = load_table('product_category_name_translation')
    sellers_df = load_table('sellers')
    geolocation_df = load_table('geolocation')
    leads_qualified_df = load_table('leads_qualified')
    leads_closed_df = load_table('leads_closed')

    # Basic preprocessing date columns
    orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
    orders_df['order_approved_at'] = pd.to_datetime(orders_df['order_approved_at'])
    orders_df['order_delivered_carrier_date'] = pd.to_datetime(orders_df['order_delivered_carrier_date'])
    orders_df['order_delivered_customer_date'] = pd.to_datetime(orders_df['order_delivered_customer_date'])
    orders_df['order_estimated_delivery_date'] = pd.to_datetime(orders_df['order_estimated_delivery_date'])

    order_items_df['shipping_limit_date'] = pd.to_datetime(order_items_df['shipping_limit_date'])

    order_reviews_df['review_creation_date'] = pd.to_datetime(order_reviews_df['review_creation_date'])
    order_reviews_df['review_answer_timestamp'] = pd.to_datetime(order_reviews_df['review_answer_timestamp'])

    leads_qualified_df['first_contact_date'] = pd.to_datetime(leads_qualified_df['first_contact_date'])
    leads_closed_df['won_date'] = pd.to_datetime(leads_closed_df['won_date'])

    # Basic preprocessing categorical columns
    product_category_name_translation_df['product_category_name'] = product_category_name_translation_df['product_category_name'].str.upper().str.replace('_', ' ')
    orders_df['order_status'] = orders_df['order_status'].str.upper()
    order_payments_df['payment_type'] = order_payments_df['payment_type'].str.upper().str.replace('_', ' ')
    products_df['product_category_name'] = products_df['product_category_name'].str.upper().str.replace('_', ' ')
    sellers_df['seller_city'] = sellers_df['seller_city'].str.upper()
    geolocation_df['geolocation_city'] = geolocation_df['geolocation_city'].str.upper()
    leads_qualified_df['origin'] = leads_qualified_df['origin'].fillna('UNKNOWN').str.upper().str.replace('_', ' ')
    leads_closed_df['business_segment'] = leads_closed_df['business_segment'].fillna('UNKNOWN').str.upper().str.replace('_', ' ')
    leads_closed_df['lead_type'] = leads_closed_df['lead_type'].fillna('UNKNOWN').str.upper().str.replace('_', ' ')
    
    return orders_df, order_payments_df, customers_df, products_df, order_items_df, order_reviews_df, product_category_name_translation_df, sellers_df, geolocation_df, leads_qualified_df, leads_closed_df

# Helper functions not in key_metrics
def temporal_trends_analysis_simple(orders_df):
    """Simple temporal trends analysis"""
    if orders_df.empty:
        return pd.DataFrame()
    
    orders_df = orders_df.copy()
    orders_df['order_month'] = pd.to_datetime(orders_df['order_purchase_timestamp']).dt.to_period('M')
    
    monthly_trends = orders_df.groupby('order_month').agg({
        'order_id': 'nunique'
    }).reset_index()
    
    monthly_trends.columns = ['order_month', 'total_orders']
    return monthly_trends

def delivery_time_analysis_simple(order_items_df, orders_df):
    """Simple delivery time analysis"""
    delivery_data = order_items_df.merge(
        orders_df[['order_id', 'order_purchase_timestamp', 'order_delivered_customer_date']], 
        on='order_id', 
        how='left'
    )
    
    delivery_data['delivery_time'] = (
        pd.to_datetime(delivery_data['order_delivered_customer_date']) - 
        pd.to_datetime(delivery_data['order_purchase_timestamp'])
    ).dt.total_seconds() / 86400  # Convert to days
    
    delivery_data = delivery_data[
        (delivery_data['delivery_time'] > 0) & 
        (delivery_data['delivery_time'] < 180)
    ]
    
    if delivery_data.empty:
        return pd.DataFrame({'avg_delivery_time_days': []})
    
    delivery_metrics = pd.DataFrame({
        'avg_delivery_time_days': delivery_data['delivery_time']
    })
    
    return delivery_metrics

# Function to apply contextual filters
def apply_contextual_filters(df, date_column=None, start_date=None, end_date=None):
    """Apply contextual filters to a DataFrame"""
    filtered_df = df.copy()
    
    # Date filter
    if date_column and date_column in filtered_df.columns and start_date and end_date:
        filtered_df[date_column] = pd.to_datetime(filtered_df[date_column])
        filtered_df = filtered_df[
            (filtered_df[date_column].dt.date >= start_date) & 
            (filtered_df[date_column].dt.date <= end_date)
        ]
    
    return filtered_df

# Load all data
orders, order_payments, customers, products, order_items, order_reviews, product_category_name_translation, sellers, geolocation, leads_qualified, leads_closed = load_data()

# ========== ENHANCED SIDEBAR FILTERS ==========
st.sidebar.header("🔍 Advanced Filters")

# Date filter with columns for better layout
st.sidebar.subheader("📅 Date Range")
min_date = orders['order_purchase_timestamp'].min().date()
max_date = orders['order_purchase_timestamp'].max().date()

col_date1, col_date2 = st.sidebar.columns(2)
with col_date1:
    start_date = st.date_input(
        "Start Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
        key="start_date"
    )
with col_date2:
    end_date = st.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        key="end_date"
    )

date_range = [start_date, end_date]

# Merge products with translations first for category filter
products_with_english = products.merge(product_category_name_translation, on='product_category_name', how='left')

# State filter
st.sidebar.subheader("📍 Location")
available_states = sorted(customers['customer_state'].unique())
customer_state = st.sidebar.multiselect(
    "Customer State",
    options=available_states,
    default=None,
    key="customer_states"
)

# Payment method filter
st.sidebar.subheader("💳 Payment")
available_payment_methods = sorted(order_payments['payment_type'].unique())
payment_method = st.sidebar.multiselect(
    "Payment Method",
    options=available_payment_methods,
    default=None,
    key="payment_methods"
)

# Product category filter
st.sidebar.subheader("📦 Products")
available_categories = sorted(products_with_english['product_category_name'].dropna().unique())
product_category = st.sidebar.multiselect(
    "Product Category",
    options=available_categories,
    default=None,
    key="product_categories",
    help="Select up to 10 categories for better performance"
)

# Apply Filters button
st.sidebar.markdown("---")
apply_filters_button = st.sidebar.button("🔄 Apply Filters", type="primary", use_container_width=True)

# Reset Filters button
if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
    st.rerun()

# ========== APPLY FILTERS TO DATA ==========
# Apply date filter to orders
filtered_orders = orders[
    (orders['order_purchase_timestamp'] >= pd.to_datetime(date_range[0])) &
    (orders['order_purchase_timestamp'] <= pd.to_datetime(date_range[1]))
].copy()

# Filter by customer state if selected
if customer_state:
    filtered_customers = customers[customers['customer_state'].isin(customer_state)]
    filtered_orders = filtered_orders[filtered_orders['customer_id'].isin(filtered_customers['customer_id'])]
else:
    filtered_customers = customers

# Filter order payments by payment method if selected
filtered_payments = order_payments.copy()
if payment_method:
    filtered_payments = order_payments[order_payments['payment_type'].isin(payment_method)]

# Filter by product category if selected
filtered_order_items = order_items.copy()
if product_category:
    filtered_products = products_with_english[products_with_english['product_category_name'].isin(product_category)]
    filtered_order_items = order_items[order_items['product_id'].isin(filtered_products['product_id'])]
    filtered_orders = filtered_orders[filtered_orders['order_id'].isin(filtered_order_items['order_id'].unique())]

# Apply date filter to leads
filtered_leads_qualified = apply_contextual_filters(
    leads_qualified, 
    date_column='first_contact_date',
    start_date=start_date,
    end_date=end_date
)

# Title
st.title("📊 Olist E-commerce Dashboard")

# SECTION 1: KEY METRICS (GENERAL)
col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

try:
    # Filter reviews based on filtered orders
    filtered_order_reviews = order_reviews[order_reviews['order_id'].isin(filtered_orders['order_id'])]
    
    # Call the function with filtered data
    satisfaction_summary, review_insights, top_narratives = customer_satisfaction_analysis(
        filtered_order_reviews,  # Use filtered reviews
        orders_df=filtered_orders,  # Use filtered orders
        products_df=products_with_english,
        order_items_df=filtered_order_items,  # Use filtered order items
        date_range=date_range
    )
except Exception as e:
    st.warning(f"Unable to load satisfaction analysis: {str(e)}")
    # Create default values if analysis fails
    review_insights = {
        'avg_score': 0,
        'nps_score': 0,
        'total_reviews': 0
    }
    filtered_order_reviews = order_reviews[order_reviews['order_id'].isin(filtered_orders['order_id'])]

with col1:
    revenue = filtered_payments[filtered_payments['order_id'].isin(filtered_orders['order_id'])]['payment_value'].sum()
    st.metric(
        label="💰 Revenue",
        value=f"${format_large_number(revenue)}",
        delta=None
    )

with col2:
    total_orders = filtered_orders['order_id'].nunique()
    st.metric(
        label="📦 Total Orders",
        value=f"{format_large_number(total_orders)}",
        delta=None
    )

with col3:
    avg_price = filtered_payments[filtered_payments['order_id'].isin(filtered_orders['order_id'])]['payment_value'].mean()
    st.metric(
        label="💸 Average Order Value",
        value=f"${avg_price:.2f}" if not pd.isna(avg_price) else "$0.00",
        delta=None
    )

with col4:
    # Calculate success rate (delivered orders)
    delivered_orders = filtered_orders[filtered_orders['order_status'] == 'DELIVERED']['order_id'].nunique()
    success_rate = (delivered_orders / total_orders * 100) if total_orders > 0 else 0
    st.metric(
        label="✅ Success Rate",
        value=f"{success_rate:.1f}%",
        delta=None
    )

with col5:
    # Calculate cancellation rate
    cancelled_orders = filtered_orders[filtered_orders['order_status'] == 'CANCELED']['order_id'].nunique()
    cancel_rate = (cancelled_orders / total_orders * 100) if total_orders > 0 else 0
    st.metric(
        label="❌ Cancellation Rate",
        value=f"{cancel_rate:.1f}%",
        delta=None
    )
    
with col6:
    # Filter reviews with scores for calculations
    filtered_reviews_with_scores = filtered_order_reviews[
        filtered_order_reviews['review_score'].notna()
    ]
    
    # Calculate average score with filtered data
    if 'avg_score' in review_insights and review_insights['avg_score'] > 0:
        avg_score = review_insights.get('avg_score', 0)
    else:
        # Fallback calculation if needed
        avg_score = filtered_reviews_with_scores['review_score'].mean() if len(filtered_reviews_with_scores) > 0 else 0
    
    # Add delta comparing to overall average
    overall_avg = order_reviews['review_score'].mean() if len(order_reviews) > 0 else avg_score
    delta_score = avg_score - overall_avg if avg_score > 0 else None
    
    st.metric(
        label="⭐ Average Rating",
        value=f"{avg_score:.2f}/5.0" if avg_score > 0 else "N/A",
        delta=f"{delta_score:+.2f}" if delta_score is not None else None,
        help=f"Based on {len(filtered_reviews_with_scores)} filtered reviews"
    )

with col7:
    # Filter reviews for NPS calculation
    filtered_reviews_for_nps = filtered_order_reviews[
        filtered_order_reviews['review_score'].notna()
    ]
    
    # Calculate NPS score with filtered data
    if 'nps_score' in review_insights and review_insights.get('nps_score', 0) != 0:
        nps_score = review_insights.get('nps_score', 0)
    else:
        # Fallback NPS calculation
        if len(filtered_reviews_for_nps) > 0:
            promoters = len(filtered_reviews_for_nps[filtered_reviews_for_nps['review_score'] >= 4])
            detractors = len(filtered_reviews_for_nps[filtered_reviews_for_nps['review_score'] <= 2])
            total = len(filtered_reviews_for_nps)
            nps_score = ((promoters - detractors) / total) * 100 if total > 0 else 0
        else:
            nps_score = 0
    
    # Calculate overall NPS for comparison
    if len(order_reviews) > 0:
        reviews_with_score = order_reviews[order_reviews['review_score'].notna()]
        if len(reviews_with_score) > 0:
            all_promoters = len(reviews_with_score[reviews_with_score['review_score'] >= 4])
            all_detractors = len(reviews_with_score[reviews_with_score['review_score'] <= 2])
            all_total = len(reviews_with_score)
            overall_nps = ((all_promoters - all_detractors) / all_total) * 100 if all_total > 0 else 0
            delta_nps = nps_score - overall_nps if nps_score != 0 else None
        else:
            delta_nps = None
    else:
        delta_nps = None
    
    st.metric(
        label="📈 NPS Score",
        value=f"{nps_score:.1f}" if nps_score != 0 else "N/A",
        delta=f"{delta_nps:+.1f}" if delta_nps is not None else None,
        help=f"Net Promoter Score based on {len(filtered_reviews_for_nps)} reviews"
    )


st.markdown("---")

# SECTION 2: PAYMENT TYPES BREAKDOWN WITH ENHANCED VISUALIZATIONS
conversion_data = conversion_rate_by_payment_method(filtered_orders, filtered_payments)

col1, col2, col3 = st.columns(3)

with col1:
    # Enhanced Payment Distribution with Bar Chart
    st.subheader("💳 Payment Distribution")
    if not conversion_data.empty:
        # Create mini bar chart for payment distribution
        payment_fig = go.Figure(data=[
            go.Bar(
                x=conversion_data['payment_type'],
                y=conversion_data['conversion_rate'] * 100,
                text=[f"{rate*100:.1f}%<br>${val:,.0f}" 
                      for rate, val in zip(conversion_data['conversion_rate'], 
                                          conversion_data['total_payment_value'])],
                textposition='outside',
                marker_color='#3498db',
                hovertemplate='<b>%{x}</b><br>' +
                            'Conversion: %{y:.1f}%<br>' +
                            '<extra></extra>'
            )
        ])
        
        payment_fig.update_layout(
            xaxis_title="Payment Type",
            yaxis_title="Conversion Rate (%)",
            height=400,
            showlegend=False,
            margin=dict(t=20, b=40, l=40, r=20)
        )
        
        st.plotly_chart(payment_fig, use_container_width=True)
    else:
        st.warning("No payment data for selected filters")

with col2:
    # Lead source distribution with filter info
    st.subheader("📊 Lead Sources")
    lead_dist = filtered_leads_qualified['origin'].value_counts()

    if not lead_dist.empty:
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', 
                '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#d35400']
        
        fig3 = go.Figure(data=[go.Pie(
            labels=lead_dist.index,
            values=lead_dist.values,
            marker=dict(
                colors=colors[:len(lead_dist)],
                line=dict(color='white', width=2)
            ),
            textposition='auto',
            textinfo='label+percent',
            textfont=dict(size=11, color='white'),
            hovertemplate='<b>%{label}</b><br>' +
                        'Count: %{value}<br>' +
                        'Percentage: %{percent}<br>' +
                        f'Period: {start_date.strftime("%d/%m/%Y")} - {end_date.strftime("%d/%m/%Y")}<br>' +
                        '<extra></extra>',
            rotation=90
        )])
        
        fig3.update_layout(
            title=dict(
                text=f"Lead Sources<br><sub>({len(lead_dist)} sources, {lead_dist.sum()} total)</sub>",
                font=dict(size=14, family="Arial, sans-serif", color='black'),
                x=0.5,
                xanchor='center'
            ),
            showlegend=True,
            height=500,
            margin=dict(t=80, b=20, l=20, r=20)
        )
        
        fig3.update_traces(
            texttemplate='<b>%{label}</b><br><b>%{percent}</b>'
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("No lead data for selected filters")

with col3:
    # Product categories distribution with enhanced info
    st.subheader("📦 Product Categories")
    if not filtered_order_items.empty:
        # Merge to get categories and order status
        category_status = filtered_order_items.merge(
            products_with_english[['product_id', 'product_category_name']], 
            on='product_id', 
            how='left'
        ).merge(
            filtered_orders[['order_id', 'order_status']], 
            on='order_id', 
            how='left'
        )
        
        # Group by category and status
        category_status_pivot = category_status.groupby(
            ['product_category_name', 'order_status']
        )['order_id'].nunique().unstack(fill_value=0)
        
        # Get top 10 categories by total orders
        category_totals = category_status_pivot.sum(axis=1).sort_values(ascending=False).head(10)
        top_categories = category_status_pivot.loc[category_totals.index]
        
        if not top_categories.empty:
            colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6', 
                    '#1abc9c', '#34495e', '#e67e22']
            
            fig2 = go.Figure()
            
            for i, col in enumerate(top_categories.columns):
                fig2.add_trace(go.Bar(
                    name=col,
                    x=top_categories.index,
                    y=top_categories[col],
                    marker_color=colors[i % len(colors)],
                    text=top_categories[col],
                    textposition='inside',
                    texttemplate='%{text:.0f}',
                    hovertemplate='<b>%{x}</b><br>' +
                                f'{col}: %{{y}} orders<br>' +
                                '<extra></extra>'
                ))
            
            total_category_orders = category_status['order_id'].nunique()
            fig2.update_layout(
                title=dict(
                    text=f"Top 10 Categories<br><sub>(Total: {total_category_orders} orders)</sub>",
                    x=0.5,
                    xanchor='center'
                ),
                xaxis=dict(
                    title="Product Category",
                    tickangle=-45,
                    tickmode='linear'
                ),
                yaxis=dict(
                    title="Number of Orders",
                ),
                barmode='stack',
                legend=dict(
                    title=dict(text="Order Status"),
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                ),
                height=500,
                margin=dict(r=150, b=100, t=80),
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No category data for selected filters")
    else:
        st.info("No order items data for selected filters")
            
st.markdown("---")

# SECTION 6: DETAILED INSIGHTS (unchanged)
st.header("📈 Detailed Performance Insights")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Temporal Trends & Forecasting",  
                            # "🚚 Delivery Performance",
                            "🔄 Cohort Analysis", 
                            "👥 RFM Segmentation",
                            "💰 Customer Lifetime Value"
                            ])

with tab1:
    # load data
    prophet_data = pd.merge(orders, order_payments, on='order_id')
    daily_revenue = prophet_data.groupby(prophet_data['order_purchase_timestamp'].dt.date)['payment_value'].sum().reset_index()
    daily_revenue.columns = ['ds', 'y']  # Prophet requiere estas columnas

    # Agregar funcionalidad Prophet
    add_prophet_tab(
        df=daily_revenue,
        date_column='ds',
        value_column='y'
    )

# with tab2:
#     # Delivery performance using simple function
#     delivery_metrics = delivery_time_analysis_simple(order_items, filtered_orders)
#     if not delivery_metrics.empty:
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             avg_delivery = delivery_metrics['avg_delivery_time_days'].mean()
#             st.metric("⏱️ Avg Delivery Time", f"{avg_delivery:.1f} days")
        
#         with col2:
#             min_delivery = delivery_metrics['avg_delivery_time_days'].min()
#             st.metric("🚀 Fastest Delivery", f"{min_delivery:.1f} days")
        
#         with col3:
#             max_delivery = delivery_metrics['avg_delivery_time_days'].max()
#             st.metric("🐌 Slowest Delivery", f"{max_delivery:.1f} days")
        
#         # Delivery time distribution
#         fig7, ax7 = plt.subplots(figsize=(10, 5))
#         ax7.hist(delivery_metrics['avg_delivery_time_days'].dropna(), bins=20, 
#                 color='#3498db', edgecolor='black', alpha=0.7)
#         ax7.axvline(avg_delivery, color='red', linestyle='--', linewidth=2, 
#                    label=f'Average: {avg_delivery:.1f} days')
#         ax7.set_xlabel("Delivery Time (days)", fontsize=11)
#         ax7.set_ylabel("Frequency", fontsize=11)
#         ax7.set_title("Delivery Time Distribution", fontsize=12, fontweight='bold')
#         ax7.legend()
#         ax7.grid(True, alpha=0.3)
#         plt.tight_layout()
#         st.pyplot(fig7)
#     else:
#         st.info("No delivery data available for the selected period")

with tab2:
    st.subheader("Customer Retention Cohort Analysis")

    # Add description
    with st.expander("ℹ️ What is Cohort Analysis?"):
        st.write("""
        **Cohort Analysis** helps understand customer retention patterns by grouping customers 
        based on their first purchase date and tracking their behavior over time.
        
        **Key Benefits:**
        - 📊 Track customer retention over time
        - 💰 Measure revenue retention by cohort
        - 🎯 Identify successful acquisition periods
        - 📈 Predict future customer behavior
        - 🔍 Discover retention patterns and trends
        """)
    
    # Quick config in columns
    col1, col2 = st.columns(2)
    with col1:
        cohort_period = st.radio(
            "Period",
            options=['month', 'quarter'],
            horizontal=True
        )
    with col2:
        retention_type = st.radio(
            "Metric",
            options=['customers', 'revenue'],
            format_func=lambda x: 'Customers' if x == 'customers' else 'Revenue',
            horizontal=True
        )
    
    # Run analysis
    if st.button("Analyze Cohorts"):
        with st.spinner("Generating cohort analysis..."):
            results = cohort_analysis_dashboard(
                orders_df=filtered_orders,
                order_payments_df=filtered_payments,
                customers_df=customers,
                date_range=date_range,
                cohort_period=cohort_period,
                retention_type=retention_type,
                max_periods=12
            )
            
            if results:
                display_cohort_analysis_streamlit(results)

with tab3:
    display_rfm_analysis_tab(
        orders_df=filtered_orders,
        order_payments_df=filtered_payments,
        customers_df=customers,
        date_range=date_range,
        customer_state=customer_state,  # From sidebar filters
        payment_method=payment_method,  # From sidebar filters  
        product_category=product_category  # From sidebar filters
    )   

with tab4:
    display_clv_analysis_tab(filtered_orders, filtered_payments, customers, 
                            date_range, customer_state, payment_method, product_category)