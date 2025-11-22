"""Business analytics functions for e-commerce data analysis"""

import pandas as pd
import warnings

# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)


def conversion_rate_by_payment_method(orders_df, order_payments_df):
    """
    Calculates the conversion rate by payment method.
    Business question: Which payment methods convert better into sales?
    Analysis: Identifies which payment methods generate more successful orders
    relative to total orders. This helps prioritize more effective payment methods
    and detect those with low acceptance.
    """
    orders_by_method = order_payments_df.groupby('payment_type')['order_id'].nunique()
    total_orders = orders_df['order_id'].nunique()
    conversion_rate = (orders_by_method / total_orders).reset_index()
    conversion_rate.columns = ['payment_type', 'conversion_rate']

    summary = order_payments_df.groupby('payment_type')['payment_value'].sum().reset_index()
    summary.columns = ['payment_type', 'total_payment_value']
    conversion_rate = conversion_rate.merge(summary, on='payment_type', how='left')
    return conversion_rate.sort_values(by='conversion_rate', ascending=False)

# --------- analisis de productos devueltos ---------

def average_value_vs_installments(order_payments_df):
    """
    Calculates the average transaction value according to the number of installments.
    Business question: How does the number of installments affect the average purchase value?
    Analysis: Helps understand if customers who pay in more installments tend
    to spend more or less, which helps design financing strategies.
    """
    summary = order_payments_df.groupby('payment_installments')['payment_value'].mean().reset_index()
    summary.columns = ['payment_installments', 'avg_transaction_value']
    return summary.sort_values(by='payment_installments')

def delivery_time_analysis(order_items_df, orders_df):
    """
    Calculates the average delivery time by product.
    Business question: How long on average do products take to be delivered?
    Analysis: Identifies products with longer delivery times,
    optimizes logistics and improves customer satisfaction.
    """
    merged_df = order_items_df.merge(orders_df[['order_id', 'order_purchase_timestamp']], on='order_id', how='left')
    merged_df['delivery_time'] = (merged_df['shipping_limit_date'] - merged_df['order_purchase_timestamp']).dt.days
    summary = merged_df.groupby('product_id')['delivery_time'].mean().reset_index()
    summary.columns = ['product_id', 'avg_delivery_time_days']
    return summary.sort_values(by='avg_delivery_time_days')

#--------- analisis de clientes RFM (reemplazar) ---------
def frequent_customers_analysis(orders_df, customers_df):
    """
    Identifies frequent customers with more than 5 orders.
    Business question: Who are the most valuable customers by recurrence?
    Analysis: Allows segmenting VIP customers and designing loyalty
    and reward strategies to increase their loyalty.
    """
    orders_per_customer = orders_df.groupby('customer_id')['order_id'].nunique().reset_index()
    orders_per_customer.columns = ['customer_id', 'total_orders']
    frequent_customers = orders_per_customer[orders_per_customer['total_orders'] > 5]
    summary = frequent_customers.merge(customers_df, on='customer_id', how='left')
    return summary.sort_values(by='total_orders', ascending=False)

def temporal_trends_analysis(orders_df):
    """
    Analyzes the number of orders by month.
    Business question: How do sales evolve over time?
    Analysis: Detects seasonal trends, demand peaks and
    plan marketing campaigns in the strongest months.
    """
    orders_df['order_month'] = orders_df['order_purchase_timestamp'].dt.to_period('M')
    summary = orders_df.groupby('order_month')['order_id'].nunique().reset_index()
    summary.columns = ['order_month', 'total_orders']
    return summary.sort_values(by='order_month')

#--------- analisis de reviews (resumen comentarios, word clod) ---------
def customer_satisfaction_analysis(order_reviews_df):
    """
    Analyzes the distribution of reviews by score.
    Business question: What is the customer satisfaction level?
    Analysis: Measures the quality of service and products, identifies
    areas for improvement and correlates satisfaction with sales.
    """
    summary = order_reviews_df.groupby('review_score')['order_id'].nunique().reset_index()
    summary.columns = ['review_score', 'total_reviews']
    return summary.sort_values(by='review_score', ascending=False)

def promotions_effectiveness_analysis(orders_df, order_payments_df, order_items_df, sellers_df, products_df):
    """
    Calculates the effectiveness rate of promotions (vouchers) and generates detailed analyses.
    Business question: How effective are promotions in generating sales and which sellers/categories work better?
    Analysis:
        - Evaluates whether promotions drive overall conversion.
        - Identifies which sellers make the most of promotions.
        - Shows which product categories benefit most from promotions.
    
    Returns:
        - effectiveness_rate (float): proportion of orders with promotion over total.
        - promotions_by_seller (DataFrame): number of orders with promotion and effectiveness rate by seller.
        - promotions_by_category (DataFrame): number of orders with promotion and effectiveness rate by product category.
    """
    # Filter payments with promotions
    promotions = order_payments_df[order_payments_df['payment_type'] == 'voucher']
    orders_with_promotion = orders_df[orders_df['order_id'].isin(promotions['order_id'])]

    # Global effectiveness rate
    effectiveness_rate = len(orders_with_promotion) / len(orders_df)

    # --- Promotions by seller ---
    promotion_items = order_items_df[order_items_df['order_id'].isin(promotions['order_id'])]
    seller_items = order_items_df.merge(sellers_df, on='seller_id', how='left')

    # Total orders by seller
    total_by_seller = seller_items.groupby('seller_id')['order_id'].nunique().reset_index()
    total_by_seller.columns = ['seller_id', 'total_orders']

    # Orders with promotion by seller
    promo_by_seller = promotion_items.merge(sellers_df, on='seller_id', how='left')
    promo_by_seller = promo_by_seller.groupby('seller_id')['order_id'].nunique().reset_index()
    promo_by_seller.columns = ['seller_id', 'promo_orders_count']

    # Merge and rate calculation
    promotions_by_seller = promo_by_seller.merge(total_by_seller, on='seller_id', how='left')
    promotions_by_seller['promo_effectiveness_rate'] = promotions_by_seller['promo_orders_count'] / promotions_by_seller['total_orders']
    promotions_by_seller = promotions_by_seller.sort_values(by='promo_effectiveness_rate', ascending=False)

    # --- Promotions by category ---
    category_items = order_items_df.merge(products_df[['product_id', 'product_category_name']], on='product_id', how='left')

    # Total orders by category
    total_by_category = category_items.groupby('product_category_name')['order_id'].nunique().reset_index()
    total_by_category.columns = ['product_category_name', 'total_orders']

    # Orders with promotion by category
    promo_by_category = promotion_items.merge(products_df[['product_id', 'product_category_name']], on='product_id', how='left')
    promo_by_category = promo_by_category.groupby('product_category_name')['order_id'].nunique().reset_index()
    promo_by_category.columns = ['product_category_name', 'promo_orders_count']

    # Merge and rate calculation
    promotions_by_category = promo_by_category.merge(total_by_category, on='product_category_name', how='left')
    promotions_by_category['promo_effectiveness_rate'] = promotions_by_category['promo_orders_count'] / promotions_by_category['total_orders']
    promotions_by_category = promotions_by_category.sort_values(by='promo_effectiveness_rate', ascending=False)

    return effectiveness_rate, promotions_by_seller, promotions_by_category


def cross_selling_analysis(order_items_df, products_df):
    """
    Analyzes the diversity of categories purchased per order.
    Business question: How many different categories do customers buy in a single order?
    Analysis: Measures cross-selling potential and design strategies
    for bundles or product recommendations.
    """
    merged_df = order_items_df.merge(products_df[['product_id', 'product_category_name']], on='product_id', how='left')
    categories_per_order = merged_df.groupby('order_id')['product_category_name'].nunique().reset_index()
    categories_per_order.columns = ['order_id', 'num_categories']
    summary = categories_per_order['num_categories'].value_counts().reset_index()
    summary.columns = ['num_categories', 'total_orders']
    return summary.sort_values(by='num_categories')

# --------- mejorarlo para que sea dinamico segun categoria, rango de fechas y estado del cliente ---------
def state_impact_analysis(customers_df, orders_df):
    """
    Analyzes the impact of geographic location on orders.
    Business question: Which states generate more sales?
    Analysis: Identifies strategic regions, optimizes local campaigns
    and adjusts logistics according to demand.
    """
    merged_df = orders_df.merge(customers_df[['customer_id', 'customer_state']], on='customer_id', how='left')
    summary = merged_df.groupby('customer_state')['order_id'].nunique().reset_index()
    summary.columns = ['customer_state', 'total_orders']
    return summary.sort_values(by='total_orders', ascending=False)


# --------- mejorarlo para que sea dinamico segun categoria, rango de fechas y estado del cliente ---------
def logistics_impact_analysis(order_items_df, orders_df):
    """
    Analyzes the impact of logistics according to order status.
    Business question: How does order status affect the number of processed orders?
    Analysis: Identifies bottlenecks in logistics and improve
    processes to increase operational efficiency.
    """
    merged_df = order_items_df.merge(orders_df[['order_id', 'order_status']], on='order_id', how='left')
    summary = merged_df.groupby('order_status')['order_id'].nunique().reset_index()
    summary.columns = ['order_status', 'total_orders']
    return summary.sort_values(by='total_orders', ascending=False)