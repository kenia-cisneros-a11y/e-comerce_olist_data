import streamlit as st
import pandas as pd

import sys
sys.path.append('scripts')

from utils import *
import matplotlib.pyplot as plt
from key_metrics import *

import warnings

# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)


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

    # basic preprocessing
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
    
    return orders_df, order_payments_df, customers_df, products_df, order_items_df, order_reviews_df, product_category_name_translation_df, sellers_df, geolocation_df, leads_qualified_df, leads_closed_df

orders, order_payments, customers, products, order_items, order_reviews, product_category_name_translation, sellers, geolocation, leads_qualified, leads_closed = load_data()

# Filters
st.sidebar.header("Filters")
min_date = orders['order_purchase_timestamp'].min()
max_date = orders['order_purchase_timestamp'].max()
date_range = st.sidebar.date_input("Date range", [min_date, max_date])

customer_state = st.sidebar.multiselect("Customer state", options=customers['customer_state'].unique())
payment_method = st.sidebar.multiselect("Payment method", options=order_payments['payment_type'].unique())
product_category = st.sidebar.multiselect("Product category", options=products['product_category_name'].unique())

# Apply filters
filtered_orders = orders[
    (orders['order_purchase_timestamp'] >= pd.to_datetime(date_range[0])) &
    (orders['order_purchase_timestamp'] <= pd.to_datetime(date_range[1]))
]

if customer_state:
    filtered_orders = filtered_orders.merge(customers, on='customer_id')
    filtered_orders = filtered_orders[filtered_orders['customer_state'].isin(customer_state)]

if payment_method:
    order_payments = order_payments[order_payments['payment_type'].isin(payment_method)]

if product_category:
    filtered_products = products[products['product_category_name'].isin(product_category)]
    order_items = order_items[order_items['product_id'].isin(filtered_products['product_id'])]

# Executive insights
st.title("📊 E-commerce Executive Dashboard")

col1, col2, col3, col4 = st.columns(4)
col1.metric("🛒 Total orders in period", f"{filtered_orders['order_id'].nunique():,}")
col2.metric("💳 Conversion rate", f"{conversion_rate_by_payment_method(filtered_orders, order_payments)['conversion_rate'].mean():.2%}")
col3.metric("💰 Average value", f"${order_payments['payment_value'].mean():.2f}")

# Get promotions effectiveness (returns tuple, use first element)
effectiveness_rate, _, _ = promotions_effectiveness_analysis(filtered_orders, order_payments, order_items, sellers, products)
col4.metric("🎟️ Promotions used", f"{effectiveness_rate:.2%}")

st.markdown("---")

# Visualization 1: Conversion by payment method
st.subheader("💳 Conversion rate by payment method")
conversion = conversion_rate_by_payment_method(filtered_orders, order_payments)
fig1, ax1 = plt.subplots()
ax1.bar(conversion['payment_type'], conversion['conversion_rate'], color='steelblue')
ax1.set_ylabel("Conversion rate")
ax1.set_xlabel("Payment type")
st.pyplot(fig1)

# Visualization 2: Average value vs installments
st.subheader("📉 Average transaction value vs. number of installments")
installments = average_value_vs_installments(order_payments)
fig2, ax2 = plt.subplots()
ax2.plot(installments['payment_installments'], installments['avg_transaction_value'], marker='o', color='green')
ax2.set_xlabel("Installments")
ax2.set_ylabel("Average value")
st.pyplot(fig2)

# Visualization 3: Monthly trends
st.subheader("📆 Order trends by month")
trends = temporal_trends_analysis(filtered_orders)
if not trends.empty:
    fig4, ax4 = plt.subplots()
    ax4.plot(trends['order_month'].astype(str), trends['total_orders'], marker='o', color='purple')
    ax4.set_xticklabels(trends['order_month'].astype(str), rotation=45)
    ax4.set_ylabel("Orders")
    ax4.set_xlabel("Month")
    plt.tight_layout()
    st.pyplot(fig4)
else:
    st.info("No data available for the selected period")

# Visualization 4: Customer satisfaction
st.subheader("😊 Customer satisfaction distribution")
satisfaction = customer_satisfaction_analysis(order_reviews)
if not satisfaction.empty:
    fig5, ax5 = plt.subplots()
    ax5.bar(satisfaction['review_score'], satisfaction['total_reviews'], color='gold')
    ax5.set_xlabel("Score")
    ax5.set_ylabel("Total reviews")
    st.pyplot(fig5)
else:
    st.info("No reviews available")

# Additional insights section
st.markdown("---")
st.header("🎯 Additional Insights")

# Create two columns for additional metrics
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Top 5 States by Orders")
    location_impact = state_impact_analysis(customers, filtered_orders)
    if not location_impact.empty:
        top_states = location_impact.head(5)
        fig6, ax6 = plt.subplots(figsize=(8, 4))
        ax6.bar(top_states['customer_state'], top_states['total_orders'], color='skyblue')
        ax6.set_xlabel("State")
        ax6.set_ylabel("Total Orders")
        plt.tight_layout()
        st.pyplot(fig6)
    else:
        st.info("No location data available")

with col2:
    st.subheader("🔄 Cross-selling Analysis")
    cross_sell = cross_selling_analysis(order_items, products)
    if not cross_sell.empty:
        fig7, ax7 = plt.subplots(figsize=(8, 4))
        ax7.bar(cross_sell['num_categories'], cross_sell['total_orders'], color='coral')
        ax7.set_xlabel("Number of Categories per Order")
        ax7.set_ylabel("Total Orders")
        plt.tight_layout()
        st.pyplot(fig7)
    else:
        st.info("No cross-selling data available")

# Logistics section
st.markdown("---")
st.header("🚚 Logistics & Operations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 Orders by Status")
    logistics = logistics_impact_analysis(order_items, filtered_orders)
    if not logistics.empty:
        fig8, ax8 = plt.subplots(figsize=(8, 4))
        ax8.pie(logistics['total_orders'], labels=logistics['order_status'], autopct='%1.1f%%')
        st.pyplot(fig8)
    else:
        st.info("No logistics data available")

with col2:
    st.subheader("⏱️ Delivery Time Analysis")
    delivery = delivery_time_analysis(order_items, filtered_orders)
    if not delivery.empty:
        st.metric("Average Delivery Time", f"{delivery['avg_delivery_time_days'].mean():.1f} days")
        st.metric("Fastest Delivery", f"{delivery['avg_delivery_time_days'].min():.1f} days")
        st.metric("Slowest Delivery", f"{delivery['avg_delivery_time_days'].max():.1f} days")
    else:
        st.info("No delivery data available")

# Footer
st.markdown("---")
st.caption("Dashboard generated with Streamlit | Data updated in real-time")