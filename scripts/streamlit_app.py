import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import warnings

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
    product_category_name_translation_df['product_category_name_english'] = product_category_name_translation_df['product_category_name_english'].str.upper().str.replace('_', ' ')
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

# Load all data
orders, order_payments, customers, products, order_items, order_reviews, product_category_name_translation, sellers, geolocation, leads_qualified, leads_closed = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Date filter
min_date = orders['order_purchase_timestamp'].min()
max_date = orders['order_purchase_timestamp'].max()
date_range = st.sidebar.date_input(
    "📅 Date range", 
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# State filter
customer_state = st.sidebar.multiselect(
    "📍 Customer state", 
    options=sorted(customers['customer_state'].unique()),
    default=None
)

# Payment method filter
payment_method = st.sidebar.multiselect(
    "💳 Payment method", 
    options=sorted(order_payments['payment_type'].unique()),
    default=None
)

# Product category filter - Merge products with translations first
products_with_english = products.merge(product_category_name_translation, on='product_category_name', how='left')
available_categories = products_with_english['product_category_name_english'].dropna().unique()
product_category = st.sidebar.multiselect(
    "📦 Product category", 
    options=sorted(available_categories),
    default=None
)

# Apply filters
filtered_orders = orders[
    (orders['order_purchase_timestamp'] >= pd.to_datetime(date_range[0])) &
    (orders['order_purchase_timestamp'] <= pd.to_datetime(date_range[1]))
].copy()

# Filter by customer state if selected
if customer_state:
    filtered_customers = customers[customers['customer_state'].isin(customer_state)]
    filtered_orders = filtered_orders[filtered_orders['customer_id'].isin(filtered_customers['customer_id'])]

# Filter order payments by payment method if selected
filtered_payments = order_payments.copy()
if payment_method:
    filtered_payments = order_payments[order_payments['payment_type'].isin(payment_method)]

# Filter by product category if selected
filtered_order_items = order_items.copy()
if product_category:
    filtered_products = products_with_english[products_with_english['product_category_name_english'].isin(product_category)]
    filtered_order_items = order_items[order_items['product_id'].isin(filtered_products['product_id'])]
    filtered_orders = filtered_orders[filtered_orders['order_id'].isin(filtered_order_items['order_id'].unique())]

# Title
st.title("📊 Executive E-commerce Dashboard")
st.markdown("---")

# SECTION 1: KEY METRICS (GENERAL)
st.header("🎯 Overall Performance Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

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

st.markdown("---")

# SECTION 2: PAYMENT TYPES BREAKDOWN
st.header("💳 Payment Methods Analysis")

# Use the conversion_rate_by_payment_method function from key_metrics
conversion_data = conversion_rate_by_payment_method(filtered_orders, filtered_payments)

col1, col2 = st.columns([1, 2])

with col1:
    # Payment type distribution
    st.subheader("Payment Distribution")
    if not conversion_data.empty:
        total_payment_sum = conversion_data['total_payment_value'].sum()
        for _, row in conversion_data.iterrows():
            percentage = (row['conversion_rate'] * 100)
            st.text(f"{row['payment_type']}: {percentage:.1f}% - ${row['total_payment_value']:,.2f}")

with col2:
    if not conversion_data.empty:
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        colors = plt.cm.Set3(range(len(conversion_data)))
        bars = ax1.bar(conversion_data['payment_type'], conversion_data['conversion_rate'] * 100)
        ax1.set_xlabel("Payment Type", fontsize=12)
        ax1.set_ylabel("Conversion Rate (%)", fontsize=12)
        ax1.set_title("Conversion Rate by Payment Method", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, val in zip(bars, conversion_data['conversion_rate'] * 100):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig1)
    else:
        st.info("No payment data available for the selected filters")

st.markdown("---")

# SECTION 3: PRODUCT CATEGORIES WITH ORDER STATUS
st.header("📦 Product Categories Performance")

# Prepare category data with order status
if not filtered_order_items.empty:
    # Merge to get categories and order status
    category_status = filtered_order_items.merge(
        products_with_english[['product_id', 'product_category_name_english']], 
        on='product_id', 
        how='left'
    ).merge(
        filtered_orders[['order_id', 'order_status']], 
        on='order_id', 
        how='left'
    )
    
    # Group by category and status
    category_status_pivot = category_status.groupby(
        ['product_category_name_english', 'order_status']
    )['order_id'].nunique().unstack(fill_value=0)
    
    # Get top 10 categories by total orders
    category_totals = category_status_pivot.sum(axis=1).sort_values(ascending=False).head(10)
    top_categories = category_status_pivot.loc[category_totals.index]
    
    if not top_categories.empty:
        fig2, ax2 = plt.subplots(figsize=(14, 8))
        
        # Create stacked bar chart
        top_categories.plot(
            kind='bar', 
            stacked=True, 
            ax=ax2,
            color=['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']
        )
        
        ax2.set_xlabel("Product Category", fontsize=12)
        ax2.set_ylabel("Number of Orders", fontsize=12)
        ax2.set_title("Top 10 Product Categories by Order Status", fontsize=14, fontweight='bold')
        ax2.legend(title="Order Status", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig2)
    else:
        st.info("No category data available for the selected filters")
else:
    st.info("No order items data available for the selected filters")

st.markdown("---")

# SECTION 4: LEAD SOURCES (ORGANIC VS PAID)
st.header("🎯 Lead Sources Analysis")

col1, col2 = st.columns(2)

with col1:
    # Lead source distribution
    lead_dist = leads_qualified['origin'].value_counts()
    
    if not lead_dist.empty:
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#d35400']
        wedges, texts, autotexts = ax3.pie(
            lead_dist.values, 
            labels=lead_dist.index,
            autopct='%1.1f%%',
            colors=colors[:len(lead_dist)],
            startangle=90,
            textprops={'fontsize': 11}
        )
        
        ax3.set_title("Lead Sources Distribution", fontsize=14, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_weight('bold')
            autotext.set_color('white')
        
        st.pyplot(fig3)
    else:
        st.info("No lead source data available")

with col2:
    st.subheader("📊 Lead Sources Breakdown")
    if not lead_dist.empty:
        total_leads = lead_dist.sum()
        
        # Categorize as organic vs paid
        organic_keywords = ['ORGANIC', 'DIRECT', 'UNKNOWN']
        paid_keywords = ['PAID', 'SOCIAL', 'EMAIL', 'REFERRAL']
        
        organic_count = sum([lead_dist.get(key, 0) for key in lead_dist.index if any(word in key for word in organic_keywords)])
        paid_count = sum([lead_dist.get(key, 0) for key in lead_dist.index if any(word in key for word in paid_keywords)])
        other_count = total_leads - organic_count - paid_count
        
        st.metric("🌱 Organic Leads", f"{organic_count:,} ({organic_count/total_leads*100:.1f}%)")
        st.metric("💵 Paid/Marketing Leads", f"{paid_count:,} ({paid_count/total_leads*100:.1f}%)")
        if other_count > 0:
            st.metric("❓ Other Sources", f"{other_count:,} ({other_count/total_leads*100:.1f}%)")
        
        st.info(f"Total Qualified Leads: {total_leads:,}")
    else:
        st.info("No lead data available")

st.markdown("---")

# SECTION 5: CUSTOMER SATISFACTION
st.header("😊 Customer Satisfaction Analysis")

# Use the customer_satisfaction_analysis function from key_metrics
try:
    # Call the function with available parameters
    satisfaction_summary, review_insights, top_narratives = customer_satisfaction_analysis(
        order_reviews, 
        orders_df=orders,
        products_df=products_with_english,
        order_items_df=order_items,
        date_range=date_range
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_score = review_insights.get('avg_score', 0)
        st.metric(
            label="⭐ Average Rating",
            value=f"{avg_score:.2f}/5.0",
            delta=None
        )
        
        # Create mini rating distribution
        if not satisfaction_summary.empty:
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            
            # Show satisfaction levels grouped
            satisfied = satisfaction_summary[satisfaction_summary['review_score'].isin([4, 5])]['percentage'].sum()
            neutral = satisfaction_summary[satisfaction_summary['review_score'] == 3]['percentage'].sum() if 3 in satisfaction_summary['review_score'].values else 0
            dissatisfied = satisfaction_summary[satisfaction_summary['review_score'].isin([1, 2])]['percentage'].sum()
            
            levels = ['Satisfied', 'Neutral', 'Dissatisfied']
            percentages = [satisfied, neutral, dissatisfied]
            colors = ['#2ecc71', '#f39c12', '#e74c3c']
            
            bars = ax4.bar(levels, percentages, color=colors)
            
            ax4.set_ylabel("Percentage (%)", fontsize=10)
            ax4.set_title("Satisfaction Distribution", fontsize=12, fontweight='bold')
            ax4.set_ylim(0, 100)
            
            # Add percentage labels
            for bar, val in zip(bars, percentages):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}%',
                        ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig4)
    
    with col2:
        satisfaction_rate = review_insights.get('satisfaction_rate', 0)
        st.metric(
            label="👍 Satisfaction Rate",
            value=f"{satisfaction_rate:.1f}%",
            delta=None
        )
        
        nps_score = review_insights.get('nps_score', 0)
        st.metric(
            label="📈 NPS Score",
            value=f"{nps_score:.1f}",
            delta=None
        )
        
        total_reviews = review_insights.get('total_reviews', 0)
        st.metric(
            label="💬 Total Reviews",
            value=f"{total_reviews:,}",
            delta=None
        )
    
    with col3:
        # Review score distribution chart
        if not satisfaction_summary.empty:
            fig5, ax5 = plt.subplots(figsize=(6, 4))
            
            review_dist = satisfaction_summary.groupby('review_score')['total_reviews'].sum().sort_index()
            
            colors = ['#e74c3c', '#e67e22', '#f39c12', '#3498db', '#2ecc71']
            bars = ax5.bar(review_dist.index, review_dist.values, color=colors)
            
            ax5.set_xlabel("Review Score", fontsize=10)
            ax5.set_ylabel("Number of Reviews", fontsize=10)
            ax5.set_title("Review Score Distribution", fontsize=12, fontweight='bold')
            ax5.set_xticks([1, 2, 3, 4, 5])
            
            # Add count labels
            for bar, val in zip(bars, review_dist.values):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(val):,}',
                        ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig5)

except Exception as e:
    st.warning(f"Unable to load satisfaction analysis: {str(e)}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("⭐ Average Rating", "N/A")
    with col2:
        st.metric("👍 Satisfaction Rate", "N/A")
    with col3:
        st.metric("💬 Total Reviews", "N/A")

st.markdown("---")

# SECTION 6: DETAILED INSIGHTS
st.header("📈 Detailed Performance Insights")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Temporal Trends", 
                            "🚚 Delivery Performance",
                            "🔄 Cohort Analysis", 
                            "👥 RFM Segmentation",
                            "💰 Lifetime Value",
                            "💡 Key Recommendations",
                            ])

with tab1:
    # Monthly trends using simple function
    trends = temporal_trends_analysis_simple(filtered_orders)
    if not trends.empty:
        fig6, ax6 = plt.subplots(figsize=(12, 6))
        
        # Orders trend
        ax6.plot(trends['order_month'].astype(str), trends['total_orders'], 
                 marker='o', color='#3498db', linewidth=2, markersize=8)
        ax6.set_xlabel("Month", fontsize=11)
        ax6.set_ylabel("Number of Orders", fontsize=11)
        ax6.set_title("Monthly Order Trends", fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        st.pyplot(fig6)
    else:
        st.info("No temporal data available for the selected period")

with tab2:
    # Delivery performance using simple function
    delivery_metrics = delivery_time_analysis_simple(order_items, filtered_orders)
    if not delivery_metrics.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_delivery = delivery_metrics['avg_delivery_time_days'].mean()
            st.metric("⏱️ Avg Delivery Time", f"{avg_delivery:.1f} days")
        
        with col2:
            min_delivery = delivery_metrics['avg_delivery_time_days'].min()
            st.metric("🚀 Fastest Delivery", f"{min_delivery:.1f} days")
        
        with col3:
            max_delivery = delivery_metrics['avg_delivery_time_days'].max()
            st.metric("🐌 Slowest Delivery", f"{max_delivery:.1f} days")
        
        # Delivery time distribution
        fig7, ax7 = plt.subplots(figsize=(10, 5))
        ax7.hist(delivery_metrics['avg_delivery_time_days'].dropna(), bins=20, 
                color='#3498db', edgecolor='black', alpha=0.7)
        ax7.axvline(avg_delivery, color='red', linestyle='--', linewidth=2, 
                   label=f'Average: {avg_delivery:.1f} days')
        ax7.set_xlabel("Delivery Time (days)", fontsize=11)
        ax7.set_ylabel("Frequency", fontsize=11)
        ax7.set_title("Delivery Time Distribution", fontsize=12, fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig7)
    else:
        st.info("No delivery data available for the selected period")

with tab3:
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

with tab4:
    display_rfm_analysis_tab(
        orders_df=filtered_orders,
        order_payments_df=filtered_payments,
        customers_df=customers,
        date_range=date_range,
        customer_state=customer_state,  # From sidebar filters
        payment_method=payment_method,  # From sidebar filters  
        product_category=product_category  # From sidebar filters
    )   

with tab5:
    display_clv_analysis_tab(filtered_orders, filtered_payments, customers, 
                            date_range, customer_state, payment_method, product_category)

with tab6:
    st.subheader("🎯 Strategic Recommendations")
    
    # Generate insights based on metrics
    insights = []
    
    if success_rate < 95:
        insights.append("⚠️ **Improve Order Fulfillment**: Success rate is below 95%. Focus on reducing cancellations and improving delivery processes.")
    
    if cancel_rate > 5:
        insights.append("🔴 **High Cancellation Rate**: Investigate reasons for cancellations and implement preventive measures.")
    
    try:
        if avg_score < 4.0:
            insights.append("⭐ **Customer Satisfaction Alert**: Average rating is below 4.0. Implement quality improvement initiatives.")
        
        if nps_score < 50:
            insights.append("📊 **NPS Improvement Needed**: Focus on converting detractors to promoters through better customer experience.")
    except:
        pass
    
    # Payment insights
    if not conversion_data.empty:
        top_payment = conversion_data.iloc[0]['payment_type']
        insights.append(f"💳 **Payment Optimization**: {top_payment} is the most used payment method. Ensure smooth processing for this channel.")
    
    # Promotions effectiveness
    try:
        effectiveness_rate, _, _ = promotions_effectiveness_analysis(
            filtered_orders, 
            filtered_payments, 
            filtered_order_items, 
            sellers, 
            products_with_english,
            product_category_name_translation
        )
        if effectiveness_rate < 0.05:
            insights.append(f"🎯 **Low Promotion Usage**: Only {effectiveness_rate:.1%} of orders use promotions. Consider more attractive promotional campaigns.")
    except:
        pass
    
    # Display insights
    if insights:
        for insight in insights:
            st.write(insight)
    else:
        st.success("✅ All key metrics are performing well! Continue monitoring for sustained success.")

# Footer
st.markdown("---")
st.caption("Dashboard generated with Streamlit | Data refreshed in real-time | Executive View")
st.caption(f"Data Period: {date_range[0]} to {date_range[1]}")


