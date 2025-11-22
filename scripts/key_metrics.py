"""Business analytics functions for e-commerce data analysis"""

import pandas as pd
import warnings
import re
from collections import Counter

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

def promotions_effectiveness_analysis(orders_df, order_payments_df, order_items_df, sellers_df, products_df, product_category_name_translation_df):
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
    products_df = products_df.merge(product_category_name_translation_df, on='product_category_name', how='left')
    category_items = order_items_df.merge(products_df[['product_id', 'product_category_name_english']], on='product_id', how='left')

    # Total orders by category
    total_by_category = category_items.groupby('product_category_name_english')['order_id'].nunique().reset_index()
    total_by_category.columns = ['product_category_name_english', 'total_orders']

    # Orders with promotion by category
    promo_by_category = promotion_items.merge(products_df[['product_id', 'product_category_name_english']], on='product_id', how='left')
    promo_by_category = promo_by_category.groupby('product_category_name_english')['order_id'].nunique().reset_index()
    promo_by_category.columns = ['product_category_name_english', 'promo_orders_count']

    # Merge and rate calculation
    promotions_by_category = promo_by_category.merge(total_by_category, on='product_category_name_english', how='left')
    promotions_by_category['promo_effectiveness_rate'] = promotions_by_category['promo_orders_count'] / promotions_by_category['total_orders']
    promotions_by_category = promotions_by_category.sort_values(by='promo_effectiveness_rate', ascending=False)

    return effectiveness_rate, promotions_by_seller, promotions_by_category


def order_status_analysis(orders_df, order_items_df=None, customers_df=None, products_df=None, 
                          date_range=None, states=None, categories=None):
    """
    Analyzes order status distribution with dynamic filters.
    Business question: What is the distribution of orders by status and how does it vary across dimensions?
    Analysis: Identifies bottlenecks in fulfillment process, cancellation/return rates, 
    and patterns by region or product category.
    
    Parameters:
    -----------
    orders_df : DataFrame - Main orders table
    order_items_df : DataFrame - Order items (optional for category filter)
    customers_df : DataFrame - Customers (optional for state filter)
    products_df : DataFrame - Products (optional for category filter)
    date_range : tuple - (start_date, end_date) for filtering
    states : list - List of states to filter
    categories : list - List of product categories to filter
    
    Returns:
    --------
    tuple: (summary_df, metrics_dict, time_series_df)
    """
    
    # Copy DataFrame to avoid modifying original
    filtered_orders = orders_df.copy()
    
    # Apply date range filter
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_orders = filtered_orders[
            (filtered_orders['order_purchase_timestamp'] >= pd.to_datetime(start_date)) &
            (filtered_orders['order_purchase_timestamp'] <= pd.to_datetime(end_date))
        ]
    
    # Apply state filter (customer location)
    if states and customers_df is not None:
        customers_filtered = customers_df[customers_df['customer_state'].isin(states)]
        filtered_orders = filtered_orders[
            filtered_orders['customer_id'].isin(customers_filtered['customer_id'])
        ]
    
    # Apply product category filter
    if categories and order_items_df is not None and products_df is not None:
        products_filtered = products_df[products_df['product_category_name_english'].isin(categories)]
        items_filtered = order_items_df[
            order_items_df['product_id'].isin(products_filtered['product_id'])
        ]
        filtered_orders = filtered_orders[
            filtered_orders['order_id'].isin(items_filtered['order_id'].unique())
        ]
    
    # 1. General summary by status
    status_summary = filtered_orders.groupby('order_status').agg({
        'order_id': 'count',
        'customer_id': 'nunique'
    }).reset_index()
    status_summary.columns = ['order_status', 'total_orders', 'unique_customers']
    
    # Calculate percentages
    total_orders = status_summary['total_orders'].sum()
    status_summary['percentage'] = (status_summary['total_orders'] / total_orders * 100).round(2)
    
    # Sort by process relevance
    status_order = ['created', 'approved', 'invoiced', 'processing', 'shipped', 
                    'delivered', 'unavailable', 'canceled']
    status_summary['sort_order'] = status_summary['order_status'].map(
        {status: i for i, status in enumerate(status_order)}
    )
    status_summary = status_summary.sort_values('sort_order').drop('sort_order', axis=1)
    
    # 2. Key metrics
    metrics = {
        'total_orders': int(total_orders),
        'delivered_rate': float(
            status_summary[status_summary['order_status'] == 'delivered']['percentage'].values[0]
            if 'delivered' in status_summary['order_status'].values else 0
        ),
        'canceled_rate': float(
            status_summary[status_summary['order_status'] == 'canceled']['percentage'].values[0]
            if 'canceled' in status_summary['order_status'].values else 0
        ),
        'in_transit': int(
            status_summary[status_summary['order_status'].isin(['shipped', 'processing'])]['total_orders'].sum()
        ),
        'pending_approval': int(
            status_summary[status_summary['order_status'] == 'created']['total_orders'].values[0]
            if 'created' in status_summary['order_status'].values else 0
        )
    }
    
    # 3. Time series by status
    time_series = filtered_orders.copy()
    time_series['order_date'] = time_series['order_purchase_timestamp'].dt.date
    
    # Group by date and status
    time_series_summary = time_series.groupby(['order_date', 'order_status'])['order_id'].count().reset_index()
    time_series_summary.columns = ['date', 'order_status', 'count']
    
    # Pivot to have statuses as columns
    time_series_pivot = time_series_summary.pivot(
        index='date', 
        columns='order_status', 
        values='count'
    ).fillna(0).reset_index()
    
    return status_summary, metrics, time_series_pivot


def order_fulfillment_funnel(orders_df, date_range=None):
    """
    Analyzes order fulfillment funnel.
    Business question: Where are orders lost in the fulfillment process?
    Analysis: Identifies critical stages where orders get stuck or canceled.
    """
    
    filtered_orders = orders_df.copy()
    
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_orders = filtered_orders[
            (filtered_orders['order_purchase_timestamp'] >= pd.to_datetime(start_date)) &
            (filtered_orders['order_purchase_timestamp'] <= pd.to_datetime(end_date))
        ]
    
    # Define ideal funnel
    funnel_stages = {
        'created': 'Created',
        'approved': 'Approved', 
        'invoiced': 'Invoiced',
        'shipped': 'Shipped',
        'delivered': 'Delivered'
    }
    
    funnel_data = []
    total_orders = len(filtered_orders)
    
    for status_key, status_name in funnel_stages.items():
        count = len(filtered_orders[filtered_orders['order_status'] == status_key])
        percentage = (count / total_orders * 100) if total_orders > 0 else 0
        funnel_data.append({
            'stage': status_name,
            'count': count,
            'percentage': percentage
        })
    
    # Add canceled as separate metric
    canceled_count = len(filtered_orders[filtered_orders['order_status'] == 'canceled'])
    canceled_percentage = (canceled_count / total_orders * 100) if total_orders > 0 else 0
    
    funnel_df = pd.DataFrame(funnel_data)
    
    return funnel_df, {'canceled_count': canceled_count, 'canceled_rate': canceled_percentage}


# ---- modificar para que se vea afectada por filtros de tiempo y categorias ---
def delivery_performance_by_state(orders_df, customers_df):
    """
    Analyzes delivery performance by state.
    Business question: Which states have best/worst delivery rates?
    Analysis: Identifies problematic regions to improve logistics.
    """
    
    # Merge with customers to get states
    orders_with_state = orders_df.merge(
        customers_df[['customer_id', 'customer_state']], 
        on='customer_id', 
        how='left'
    )
    
    # Group by state and status
    state_status = orders_with_state.groupby(['customer_state', 'order_status'])['order_id'].count().reset_index()
    state_status.columns = ['state', 'status', 'count']
    
    # Pivot for analysis
    state_pivot = state_status.pivot(index='state', columns='status', values='count').fillna(0)
    
    # Calculate key metrics by state
    state_metrics = pd.DataFrame()
    state_metrics['state'] = state_pivot.index
    state_metrics['total_orders'] = state_pivot.sum(axis=1).values
    
    # Delivery rate
    state_metrics['delivery_rate'] = (
        (state_pivot['delivered'] / state_pivot.sum(axis=1) * 100)
        if 'delivered' in state_pivot.columns else 0
    ).values.round(2)
    
    # Cancellation rate
    state_metrics['cancellation_rate'] = (
        (state_pivot['canceled'] / state_pivot.sum(axis=1) * 100)
        if 'canceled' in state_pivot.columns else 0
    ).values.round(2)
    
    # Orders in transit
    transit_cols = ['processing', 'shipped']
    state_metrics['in_transit'] = state_pivot[
        [col for col in transit_cols if col in state_pivot.columns]
    ].sum(axis=1).values if any(col in state_pivot.columns for col in transit_cols) else 0
    
    return state_metrics.sort_values('delivery_rate', ascending=False)


def rfm_customer_segmentation(orders_df, order_payments_df, customers_df=None, 
                              date_range=None, reference_date=None):
    """
    Performs RFM (Recency, Frequency, Monetary) analysis for customer segmentation.
    Business question: How can we categorize customers for targeted marketing and promotions?
    Analysis: Creates customer segments based on purchase behavior to enable personalized
    marketing strategies, retention programs, and promotional targeting.
    
    Parameters:
    -----------
    orders_df : DataFrame - Orders table
    order_payments_df : DataFrame - Payment information
    customers_df : DataFrame - Customer demographics (optional)
    date_range : tuple - (start_date, end_date) for filtering
    reference_date : datetime - Reference date for recency calculation (default: most recent date)
    
    Returns:
    --------
    tuple: (rfm_df, segment_summary, recommendations)
    """
    
    # Apply date filter if provided
    filtered_orders = orders_df.copy()
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_orders = filtered_orders[
            (filtered_orders['order_purchase_timestamp'] >= pd.to_datetime(start_date)) &
            (filtered_orders['order_purchase_timestamp'] <= pd.to_datetime(end_date))
        ]
    
    # Set reference date for recency calculation
    if reference_date is None:
        reference_date = filtered_orders['order_purchase_timestamp'].max()
    else:
        reference_date = pd.to_datetime(reference_date)
    
    # Merge with payments to get monetary value
    orders_with_value = filtered_orders.merge(
        order_payments_df.groupby('order_id')['payment_value'].sum().reset_index(),
        on='order_id',
        how='left'
    )
    
    # Calculate RFM metrics
    rfm = orders_with_value.groupby('customer_id').agg({
        'order_purchase_timestamp': lambda x: (reference_date - x.max()).days,  # Recency
        'order_id': 'nunique',  # Frequency
        'payment_value': 'sum'  # Monetary
    }).reset_index()
    
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    # Calculate RFM scores using quartiles
    rfm['r_quartile'] = pd.qcut(rfm['recency'], 4, labels=['4', '3', '2', '1'])
    rfm['f_quartile'] = pd.qcut(rfm['frequency'].rank(method='first'), 4, labels=['1', '2', '3', '4'])
    rfm['m_quartile'] = pd.qcut(rfm['monetary'], 4, labels=['1', '2', '3', '4'])
    
    # Combine RFM scores
    rfm['rfm_score'] = rfm['r_quartile'].astype(str) + rfm['f_quartile'].astype(str) + rfm['m_quartile'].astype(str)
    
    # Define customer segments based on RFM scores
    def segment_customers(row):
        r, f, m = int(row['r_quartile']), int(row['f_quartile']), int(row['m_quartile'])
        
        # Champions: Best customers
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        # Loyal Customers: High frequency, good monetary
        elif f >= 3 and m >= 3:
            return 'Loyal Customers'
        # Potential Loyalists: Recent customers with potential
        elif r >= 3 and f >= 2:
            return 'Potential Loyalists'
        # New Customers: Recent first-time buyers
        elif r >= 4 and f == 1:
            return 'New Customers'
        # At Risk: Were good customers but haven't purchased recently
        elif r == 2 and f >= 3 and m >= 3:
            return 'At Risk'
        # Can't Lose Them: Were best customers but haven't purchased recently
        elif r <= 2 and f >= 4 and m >= 4:
            return "Can't Lose Them"
        # Hibernating: Low recency, low frequency
        elif r <= 2 and f <= 2:
            return 'Hibernating'
        # Price Sensitive: Low monetary value
        elif m <= 2:
            return 'Price Sensitive'
        else:
            return 'Regular'
    
    rfm['segment'] = rfm.apply(segment_customers, axis=1)
    
    # Add customer demographics if available
    if customers_df is not None:
        rfm = rfm.merge(
            customers_df[['customer_id', 'customer_state', 'customer_city']], 
            on='customer_id', 
            how='left'
        )
    
    # Calculate segment summary
    segment_summary = rfm.groupby('segment').agg({
        'customer_id': 'count',
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean'
    }).round(2).reset_index()
    
    segment_summary.columns = ['segment', 'customer_count', 'avg_recency_days', 'avg_frequency', 'avg_monetary']
    segment_summary['percentage'] = (segment_summary['customer_count'] / len(rfm) * 100).round(2)
    
    # Sort by customer value
    segment_order = ['Champions', 'Loyal Customers', 'Potential Loyalists', 'New Customers', 
                     'Regular', 'Price Sensitive', 'At Risk', "Can't Lose Them", 'Hibernating']
    segment_summary['sort_order'] = segment_summary['segment'].map(
        {seg: i for i, seg in enumerate(segment_order)}
    )
    segment_summary = segment_summary.sort_values('sort_order').drop('sort_order', axis=1)
    
    # Generate recommendations for each segment
    recommendations = {
        'Champions': {
            'strategy': 'Reward & Retain',
            'actions': ['VIP early access to new products', 'Exclusive discounts', 'Personal account manager'],
            'discount_range': '10-15%',
            'communication': 'Premium channels, personalized content'
        },
        'Loyal Customers': {
            'strategy': 'Upsell & Appreciate',
            'actions': ['Loyalty program enrollment', 'Bundle offers', 'Referral incentives'],
            'discount_range': '10-20%',
            'communication': 'Regular engagement, product recommendations'
        },
        'Potential Loyalists': {
            'strategy': 'Engage & Convert',
            'actions': ['Membership offers', 'Product recommendations', 'Onboarding support'],
            'discount_range': '15-25%',
            'communication': 'Educational content, success stories'
        },
        'New Customers': {
            'strategy': 'Onboard & Nurture',
            'actions': ['Welcome series', 'First-purchase discount', 'Product education'],
            'discount_range': '20-30%',
            'communication': 'Onboarding emails, getting started guides'
        },
        'At Risk': {
            'strategy': 'Re-engage Urgently',
            'actions': ['Win-back campaigns', 'Special offers', 'Feedback surveys'],
            'discount_range': '25-35%',
            'communication': 'Reactivation campaigns, personalized offers'
        },
        "Can't Lose Them": {
            'strategy': 'Win Back Immediately',
            'actions': ['Direct outreach', 'Exclusive comeback offer', 'Problem resolution'],
            'discount_range': '30-40%',
            'communication': 'Personal calls, high-value incentives'
        },
        'Hibernating': {
            'strategy': 'Reactivate',
            'actions': ['Reactivation campaigns', 'New product announcements', 'Deep discounts'],
            'discount_range': '35-50%',
            'communication': 'Win-back emails, special promotions'
        },
        'Price Sensitive': {
            'strategy': 'Value Focus',
            'actions': ['Budget options', 'Volume discounts', 'Seasonal sales alerts'],
            'discount_range': '25-40%',
            'communication': 'Sale notifications, value propositions'
        },
        'Regular': {
            'strategy': 'Maintain & Grow',
            'actions': ['Regular promotions', 'Cross-sell opportunities', 'Engagement programs'],
            'discount_range': '15-25%',
            'communication': 'Standard marketing, seasonal campaigns'
        }
    }
    
    return rfm, segment_summary, recommendations


def customer_lifetime_value(orders_df, order_payments_df, customers_df=None, 
                           months_ahead=12):
    """
    Calculates Customer Lifetime Value (CLV) predictions.
    Business question: What is the predicted value of each customer segment?
    Analysis: Helps prioritize marketing spend and customer retention efforts.
    
    Parameters:
    -----------
    orders_df : DataFrame - Orders table
    order_payments_df : DataFrame - Payment information
    customers_df : DataFrame - Customer demographics (optional)
    months_ahead : int - Number of months to project ahead
    
    Returns:
    --------
    DataFrame with CLV calculations by customer
    """
    
    # Calculate historical metrics
    orders_with_value = orders_df.merge(
        order_payments_df.groupby('order_id')['payment_value'].sum().reset_index(),
        on='order_id',
        how='left'
    )
    
    # Calculate purchase intervals
    customer_purchases = orders_with_value.groupby('customer_id').agg({
        'order_purchase_timestamp': ['min', 'max', 'count'],
        'payment_value': ['sum', 'mean']
    }).reset_index()
    
    customer_purchases.columns = ['customer_id', 'first_purchase', 'last_purchase', 
                                  'purchase_count', 'total_value', 'avg_order_value']
    
    # Calculate average purchase frequency (purchases per month)
    customer_purchases['customer_age_days'] = (
        customer_purchases['last_purchase'] - customer_purchases['first_purchase']
    ).dt.days + 1
    
    customer_purchases['customer_age_months'] = customer_purchases['customer_age_days'] / 30
    customer_purchases['purchase_frequency'] = (
        customer_purchases['purchase_count'] / customer_purchases['customer_age_months']
    ).fillna(0)
    
    # Predict future value
    customer_purchases['predicted_purchases'] = (
        customer_purchases['purchase_frequency'] * months_ahead
    )
    customer_purchases['predicted_clv'] = (
        customer_purchases['predicted_purchases'] * customer_purchases['avg_order_value']
    )
    
    # Categorize CLV tiers
    def clv_tier(value):
        if value >= customer_purchases['predicted_clv'].quantile(0.8):
            return 'High Value'
        elif value >= customer_purchases['predicted_clv'].quantile(0.5):
            return 'Medium Value'
        else:
            return 'Low Value'
    
    customer_purchases['clv_tier'] = customer_purchases['predicted_clv'].apply(clv_tier)
    
    return customer_purchases


def segment_recommendation_engine(rfm_df, segment_summary, product_categories=None):
    """
    Generates specific product and promotion recommendations for each segment.
    Business question: What specific actions should we take for each customer segment?
    Analysis: Provides actionable recommendations for marketing campaigns.
    
    Parameters:
    -----------
    rfm_df : DataFrame - RFM analysis results
    segment_summary : DataFrame - Segment summary statistics
    product_categories : list - Available product categories (optional)
    
    Returns:
    --------
    DataFrame with specific recommendations per segment
    """
    
    recommendations_list = []
    
    for segment in segment_summary['segment'].unique():
        segment_data = segment_summary[segment_summary['segment'] == segment].iloc[0]
        
        rec = {
            'segment': segment,
            'customer_count': int(segment_data['customer_count']),
            'avg_recency': float(segment_data['avg_recency_days']),
            'avg_frequency': float(segment_data['avg_frequency']),
            'avg_monetary': float(segment_data['avg_monetary']),
            'priority_score': 0
        }
        
        # Calculate priority score
        if segment == 'Champions':
            rec['priority_score'] = 10
            rec['campaign_type'] = 'VIP Retention'
            rec['channel'] = 'Email + Phone'
            rec['timing'] = 'Immediate'
            
        elif segment == 'At Risk':
            rec['priority_score'] = 9
            rec['campaign_type'] = 'Win-back'
            rec['channel'] = 'Email + SMS'
            rec['timing'] = 'Within 7 days'
            
        elif segment == "Can't Lose Them":
            rec['priority_score'] = 9
            rec['campaign_type'] = 'Emergency Retention'
            rec['channel'] = 'Phone + Email'
            rec['timing'] = 'Within 3 days'
            
        elif segment == 'Loyal Customers':
            rec['priority_score'] = 8
            rec['campaign_type'] = 'Loyalty Program'
            rec['channel'] = 'Email'
            rec['timing'] = 'Within 14 days'
            
        elif segment == 'Potential Loyalists':
            rec['priority_score'] = 7
            rec['campaign_type'] = 'Engagement'
            rec['channel'] = 'Email + Push'
            rec['timing'] = 'Within 14 days'
            
        elif segment == 'New Customers':
            rec['priority_score'] = 6
            rec['campaign_type'] = 'Onboarding'
            rec['channel'] = 'Email'
            rec['timing'] = 'Immediate'
            
        else:
            rec['priority_score'] = 5
            rec['campaign_type'] = 'General'
            rec['channel'] = 'Email'
            rec['timing'] = 'Monthly'
        
        recommendations_list.append(rec)
    
    recommendations_df = pd.DataFrame(recommendations_list)
    return recommendations_df.sort_values('priority_score', ascending=False)

def customer_satisfaction_analysis(order_reviews_df, orders_df=None, products_df=None, 
                                  order_items_df=None, date_range=None, 
                                  product_categories=None):
    """
    Comprehensive customer satisfaction analysis with text analytics.
    Business question: What drives customer satisfaction and what are the main pain points?
    Analysis: Measures satisfaction levels, identifies key themes in reviews, 
    and provides actionable insights by product category.
    
    Parameters:
    -----------
    order_reviews_df : DataFrame - Reviews table
    orders_df : DataFrame - Orders table (optional, for date filtering)
    products_df : DataFrame - Products table (optional, for category filtering)
    order_items_df : DataFrame - Order items (optional, for product linking)
    date_range : tuple - (start_date, end_date) for filtering
    product_categories : list - Product categories to filter
    
    Returns:
    --------
    tuple: (satisfaction_summary, review_insights, top_narratives)
    """
    
    # Copy to avoid modifying original
    reviews = order_reviews_df.copy()
    
    # Apply date filter if orders_df is provided
    if date_range and orders_df is not None:
        start_date, end_date = date_range
        filtered_orders = orders_df[
            (orders_df['order_purchase_timestamp'] >= pd.to_datetime(start_date)) &
            (orders_df['order_purchase_timestamp'] <= pd.to_datetime(end_date))
        ]
        reviews = reviews[reviews['order_id'].isin(filtered_orders['order_id'])]
    
    # Apply product category filter
    if product_categories and all([products_df is not None, order_items_df is not None]):
        filtered_products = products_df[products_df['product_category_name_english'].isin(product_categories)]
        filtered_items = order_items_df[order_items_df['product_id'].isin(filtered_products['product_id'])]
        reviews = reviews[reviews['order_id'].isin(filtered_items['order_id'])]
    
    # 1. Satisfaction Summary
    satisfaction_summary = reviews.groupby('review_score').agg({
        'order_id': 'count',
        'review_comment_title': lambda x: x.notna().sum(),
        'review_comment_message': lambda x: x.notna().sum()
    }).reset_index()
    
    satisfaction_summary.columns = ['review_score', 'total_reviews', 'reviews_with_title', 'reviews_with_message']
    satisfaction_summary['percentage'] = (satisfaction_summary['total_reviews'] / satisfaction_summary['total_reviews'].sum() * 100).round(2)
    
    # Add satisfaction labels
    satisfaction_summary['satisfaction_level'] = satisfaction_summary['review_score'].map({
        5: 'Very Satisfied',
        4: 'Satisfied',
        3: 'Neutral',
        2: 'Dissatisfied',
        1: 'Very Dissatisfied'
    })
    
    # Calculate engagement rate by score
    satisfaction_summary['comment_rate'] = (
        satisfaction_summary['reviews_with_message'] / satisfaction_summary['total_reviews'] * 100
    ).round(2)
    
    # 2. Review Insights
    review_insights = {
        'total_reviews': len(reviews),
        'avg_score': reviews['review_score'].mean(),
        'satisfaction_rate': (reviews['review_score'] >= 4).mean() * 100,  # % of 4-5 star reviews
        'dissatisfaction_rate': (reviews['review_score'] <= 2).mean() * 100,  # % of 1-2 star reviews
        'reviews_with_comments': reviews['review_comment_message'].notna().sum(),
        'comment_rate': (reviews['review_comment_message'].notna().sum() / len(reviews) * 100) if len(reviews) > 0 else 0,
        'median_score': reviews['review_score'].median(),
        'score_std': reviews['review_score'].std()
    }
    
    # 3. NPS Calculation (Net Promoter Score)
    promoters = (reviews['review_score'] >= 4).sum()
    detractors = (reviews['review_score'] <= 2).sum()
    review_insights['nps_score'] = ((promoters - detractors) / len(reviews) * 100) if len(reviews) > 0 else 0
    
    # 4. Calculate engagement metrics
    review_insights['engaged_customers'] = reviews[
        reviews['review_comment_message'].notna() | reviews['review_comment_title'].notna()
    ]['order_id'].nunique()
    
    # 5. Text Analysis for Top Narratives
    top_narratives = analyze_review_narratives(reviews, products_df, order_items_df)
    
    return satisfaction_summary, review_insights, top_narratives


def analyze_review_narratives(reviews_df, products_df=None, order_items_df=None):
    """
    Analyzes review comments to extract key themes and narratives.
    Translates Portuguese comments to English insights.
    
    Returns:
    --------
    dict: Top narratives by satisfaction level and product category
    """
    
    # Common Portuguese-English translations for e-commerce context
    pt_en_keywords = {
        # Positive
        'bom': 'good', 'ótimo': 'excellent', 'otimo': 'excellent', 'excelente': 'excellent',
        'rápido': 'fast', 'rapido': 'fast', 'perfeito': 'perfect',
        'recomendo': 'recommend', 'satisfeito': 'satisfied', 'feliz': 'happy',
        'qualidade': 'quality', 'adorei': 'loved', 'amei': 'loved',
        'chegou': 'arrived', 'pontual': 'punctual', 'bem': 'well',
        'lindo': 'beautiful', 'linda': 'beautiful', 'bonito': 'nice',
        
        # Negative
        'ruim': 'bad', 'péssimo': 'terrible', 'pessimo': 'terrible', 'horrível': 'horrible',
        'demorou': 'delayed', 'demora': 'delay', 'atraso': 'late', 'atrasou': 'delayed',
        'defeito': 'defect', 'quebrado': 'broken', 'errado': 'wrong', 'danificado': 'damaged',
        'cancelar': 'cancel', 'cancelamento': 'cancellation', 'devolução': 'return', 'devolver': 'return',
        'problema': 'problem', 'decepção': 'disappointment', 'insatisfeito': 'unsatisfied',
        'não': 'not', 'nao': 'not', 'nunca': 'never', 'mal': 'poor',
        
        # Neutral/Descriptive
        'entrega': 'delivery', 'produto': 'product', 'embalagem': 'packaging',
        'preço': 'price', 'preco': 'price', 'valor': 'value', 'caro': 'expensive',
        'atendimento': 'service', 'vendedor': 'seller', 'frete': 'shipping',
        'prazo': 'deadline', 'tempo': 'time', 'dias': 'days'
    }
    
    narratives = {
        'positive_themes': [],
        'negative_themes': [],
        'improvement_areas': [],
        'by_category': {},
        'word_frequency': {}
    }
    
    # Separate reviews by satisfaction level
    positive_reviews = reviews_df[reviews_df['review_score'] >= 4]['review_comment_message'].dropna()
    negative_reviews = reviews_df[reviews_df['review_score'] <= 2]['review_comment_message'].dropna()
    neutral_reviews = reviews_df[reviews_df['review_score'] == 3]['review_comment_message'].dropna()
    
    # Extract themes from positive reviews
    positive_words = []
    for comment in positive_reviews:
        if isinstance(comment, str):
            comment_lower = comment.lower()
            for pt_word, en_word in pt_en_keywords.items():
                if pt_word in comment_lower:
                    positive_words.append(en_word)
    
    # Extract themes from negative reviews
    negative_words = []
    improvement_keywords = []
    for comment in negative_reviews:
        if isinstance(comment, str):
            comment_lower = comment.lower()
            for pt_word, en_word in pt_en_keywords.items():
                if pt_word in comment_lower:
                    negative_words.append(en_word)
                    # Identify improvement areas
                    if en_word in ['delayed', 'broken', 'wrong', 'defect', 'return', 'damaged', 'problem']:
                        improvement_keywords.append(en_word)
    
    # Count and rank themes
    if positive_words:
        positive_counter = Counter(positive_words)
        narratives['positive_themes'] = [
            {'theme': word, 'mentions': count, 'percentage': round(count/len(positive_reviews)*100, 2)}
            for word, count in positive_counter.most_common(7)
        ]
    
    if negative_words:
        negative_counter = Counter(negative_words)
        narratives['negative_themes'] = [
            {'theme': word, 'mentions': count, 'percentage': round(count/len(negative_reviews)*100, 2)}
            for word, count in negative_counter.most_common(7)
        ]
    
    if improvement_keywords:
        improvement_counter = Counter(improvement_keywords)
        narratives['improvement_areas'] = [
            {
                'area': word, 
                'frequency': count, 
                'priority': 'high' if count > len(negative_reviews)*0.1 else 'medium',
                'impact': 'critical' if word in ['broken', 'defect', 'damaged'] else 'moderate'
            }
            for word, count in improvement_counter.most_common(5)
        ]
    
    # Analysis by product category if available
    if products_df is not None and order_items_df is not None:
        # Link reviews to products
        reviews_with_products = reviews_df.merge(
            order_items_df[['order_id', 'product_id']], 
            on='order_id', 
            how='left'
        ).merge(
            products_df[['product_id', 'product_category_name_english']], 
            on='product_id', 
            how='left'
        )
        
        # Get top categories with issues
        category_scores = reviews_with_products.groupby('product_category_name_english').agg({
            'review_score': ['mean', 'count'],
            'review_comment_message': lambda x: x.notna().sum()
        })
        category_scores.columns = ['avg_score', 'total_reviews', 'comments_count']
        category_scores = category_scores[category_scores['total_reviews'] >= 5]  # Min 5 reviews
        
        # Find problematic categories
        problematic_categories = category_scores.nsmallest(5, 'avg_score')
        for category in problematic_categories.index:
            if pd.notna(category):
                cat_reviews = reviews_with_products[
                    reviews_with_products['product_category_name_english'] == category
                ]
                narratives['by_category'][category] = {
                    'avg_score': round(cat_reviews['review_score'].mean(), 2),
                    'total_reviews': len(cat_reviews),
                    'satisfaction_rate': round((cat_reviews['review_score'] >= 4).mean() * 100, 2),
                    'comment_rate': round(cat_reviews['review_comment_message'].notna().mean() * 100, 2),
                    'main_issues': get_category_issues(cat_reviews, pt_en_keywords)
                }
    
    # Overall word frequency
    all_comments = reviews_df['review_comment_message'].dropna()
    all_words = []
    for comment in all_comments:
        if isinstance(comment, str):
            comment_lower = comment.lower()
            for pt_word, en_word in pt_en_keywords.items():
                if pt_word in comment_lower:
                    all_words.append(en_word)
    
    if all_words:
        word_counter = Counter(all_words)
        narratives['word_frequency'] = dict(word_counter.most_common(10))
    
    return narratives


def get_category_issues(category_reviews, pt_en_keywords):
    """
    Extract main issues for a specific category.
    """
    negative_reviews = category_reviews[category_reviews['review_score'] <= 2]['review_comment_message'].dropna()
    issues = []
    
    issue_keywords = ['delayed', 'broken', 'wrong', 'defect', 'return', 'damaged', 'problem', 'poor', 'bad']
    
    for comment in negative_reviews:
        if isinstance(comment, str):
            comment_lower = comment.lower()
            for pt_word, en_word in pt_en_keywords.items():
                if pt_word in comment_lower and en_word in issue_keywords:
                    issues.append(en_word)
    
    if issues:
        issue_counter = Counter(issues)
        return [issue for issue, _ in issue_counter.most_common(3)]
    return issues


def review_sentiment_trends(reviews_df, orders_df):
    """
    Analyzes sentiment trends over time.
    Business question: How is customer sentiment evolving?
    Analysis: Tracks satisfaction trends to identify improving or declining areas.
    """
    
    # Merge with orders to get dates
    reviews_with_dates = reviews_df.merge(
        orders_df[['order_id', 'order_purchase_timestamp']], 
        on='order_id', 
        how='left'
    )
    
    # Group by month
    reviews_with_dates['review_month'] = pd.to_datetime(
        reviews_with_dates['order_purchase_timestamp']
    ).dt.to_period('M')
    
    # Calculate monthly metrics
    monthly_sentiment = reviews_with_dates.groupby('review_month').agg({
        'review_score': ['mean', 'count', 'std'],
        'review_comment_message': lambda x: x.notna().sum()
    }).round(2)
    
    monthly_sentiment.columns = ['avg_score', 'total_reviews', 'score_std', 'reviews_with_comments']
    
    # Add satisfaction rate
    monthly_sentiment['satisfaction_rate'] = reviews_with_dates.groupby('review_month').apply(
        lambda x: (x['review_score'] >= 4).mean() * 100
    ).round(2)
    
    # Add NPS by month
    monthly_sentiment['nps'] = reviews_with_dates.groupby('review_month').apply(
        lambda x: ((x['review_score'] >= 4).sum() - (x['review_score'] <= 2).sum()) / len(x) * 100
    ).round(0)
    
    return monthly_sentiment.reset_index()


def customer_cohort_retention_analysis(orders_df, customers_df=None, date_range=None, cohort_period='month'):
    """
    Performs customer cohort analysis to measure retention rates over time.
    Business question: How well are we retaining customers over time and what are the retention patterns?
    Analysis:
        - Groups customers into acquisition cohorts based on their first purchase date.
        - Tracks how many customers from each cohort return in subsequent periods.
        - Identifies retention trends and patterns across different time periods.
        - Helps evaluate long-term customer value and loyalty program effectiveness.
    
    Returns:
        - retention_matrix (DataFrame): Cohort retention matrix with percentages.
        - cohort_sizes (DataFrame): Size of each acquisition cohort.
        - retention_metrics (dict): Key retention performance indicators.
        - retention_trends (DataFrame): Monthly retention trend analysis.
    """
    
    # Filter orders by date range if provided
    filtered_orders = orders_df.copy()
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_orders = filtered_orders[
            (filtered_orders['order_purchase_timestamp'] >= pd.to_datetime(start_date)) &
            (filtered_orders['order_purchase_timestamp'] <= pd.to_datetime(end_date))
        ]
    
    # Ensure we have required columns
    required_cols = ['customer_id', 'order_purchase_timestamp']
    if not all(col in filtered_orders.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")
    
    # Create customer acquisition cohorts
    # Step 1: Identify each customer's first purchase date (cohort date)
    customer_first_purchase = filtered_orders.groupby('customer_id')['order_purchase_timestamp'].min().reset_index()
    customer_first_purchase.columns = ['customer_id', 'cohort_date']
    
    # Step 2: Assign cohort period based on specified frequency
    if cohort_period == 'month':
        customer_first_purchase['cohort_period'] = customer_first_purchase['cohort_date'].dt.to_period('M')
    elif cohort_period == 'quarter':
        customer_first_purchase['cohort_period'] = customer_first_purchase['cohort_date'].dt.to_period('Q')
    elif cohort_period == 'week':
        customer_first_purchase['cohort_period'] = customer_first_purchase['cohort_date'].dt.to_period('W')
    else:
        raise ValueError("cohort_period must be 'month', 'quarter', or 'week'")
    
    # Step 3: Merge cohort information back with orders
    orders_with_cohort = filtered_orders.merge(customer_first_purchase[['customer_id', 'cohort_period']], 
                                              on='customer_id', how='left')
    
    # Step 4: Calculate period for each order relative to cohort
    if cohort_period == 'month':
        orders_with_cohort['order_period'] = orders_with_cohort['order_purchase_timestamp'].dt.to_period('M')
    elif cohort_period == 'quarter':
        orders_with_cohort['order_period'] = orders_with_cohort['order_purchase_timestamp'].dt.to_period('Q')
    elif cohort_period == 'week':
        orders_with_cohort['order_period'] = orders_with_cohort['order_purchase_timestamp'].dt.to_period('W')
    
    # Step 5: Calculate periods since acquisition for each order
    orders_with_cohort['periods_since_acquisition'] = (
        orders_with_cohort['order_period'] - orders_with_cohort['cohort_period']
    ).apply(lambda x: x.n)
    
    # Step 6: Create cohort analysis matrix
    # Count unique customers per cohort and period
    cohort_data = orders_with_cohort.groupby(['cohort_period', 'periods_since_acquisition']).agg({
        'customer_id': 'nunique'
    }).reset_index()
    cohort_data.columns = ['cohort_period', 'periods_since_acquisition', 'customer_count']
    
    # Step 7: Pivot to create retention matrix
    retention_matrix = cohort_data.pivot_table(
        index='cohort_period',
        columns='periods_since_acquisition',
        values='customer_count',
        fill_value=0
    )
    
    # Step 8: Calculate cohort sizes (initial customer count for each cohort)
    cohort_sizes = retention_matrix.iloc[:, 0].reset_index()
    cohort_sizes.columns = ['cohort_period', 'cohort_size']
    
    # Step 9: Convert absolute counts to retention percentages
    retention_matrix_pct = retention_matrix.div(retention_matrix.iloc[:, 0], axis=0) * 100
    retention_matrix_pct = retention_matrix_pct.round(2)
    
    # Step 10: Calculate key retention metrics
    retention_metrics = calculate_retention_metrics(retention_matrix_pct, cohort_sizes)
    
    # Step 11: Analyze retention trends over time
    retention_trends = analyze_retention_trends(retention_matrix_pct)
    
    return retention_matrix_pct, cohort_sizes, retention_metrics, retention_trends


def calculate_retention_metrics(retention_matrix_pct, cohort_sizes):
    """
    Calculates key retention performance indicators from cohort analysis.
    
    Returns:
        - dict: Comprehensive retention metrics and performance indicators.
    """
    
    metrics = {}
    
    # Overall average retention rates by period
    periods = retention_matrix_pct.columns
    for period in periods[1:6]:  # Focus on first 6 periods
        if period in retention_matrix_pct.columns:
            metrics[f'period_{period}_retention_avg'] = round(retention_matrix_pct[period].mean(), 2)
    
    # First period retention (crucial metric)
    metrics['first_period_retention_avg'] = round(
        retention_matrix_pct.iloc[:, 1].mean() if len(retention_matrix_pct.columns) > 1 else 0, 2
    )
    
    # Cohort size trends
    metrics['avg_cohort_size'] = round(cohort_sizes['cohort_size'].mean(), 2)
    metrics['cohort_size_growth_rate'] = round(
        (cohort_sizes['cohort_size'].iloc[-1] - cohort_sizes['cohort_size'].iloc[0]) / 
        cohort_sizes['cohort_size'].iloc[0] * 100, 2
    ) if len(cohort_sizes) > 1 else 0
    
    # Best and worst performing cohorts
    if len(retention_matrix_pct.columns) > 1:
        best_cohort_retention = retention_matrix_pct.iloc[:, 1].max()
        worst_cohort_retention = retention_matrix_pct.iloc[:, 1].min()
        metrics['best_cohort_retention'] = round(best_cohort_retention, 2)
        metrics['worst_cohort_retention'] = round(worst_cohort_retention, 2)
        metrics['retention_variability'] = round(best_cohort_retention - worst_cohort_retention, 2)
    
    # Long-term retention (period 3+)
    if len(retention_matrix_pct.columns) >= 4:
        long_term_retention = retention_matrix_pct.iloc[:, 3].mean()
        metrics['long_term_retention_avg'] = round(long_term_retention, 2)
    
    return metrics


def analyze_retention_trends(retention_matrix_pct):
    """
    Analyzes retention trends across different cohort periods.
    
    Returns:
        - DataFrame: Monthly retention trends and performance analysis.
    """
    
    trends_data = []
    
    for cohort in retention_matrix_pct.index:
        cohort_data = {'cohort_period': str(cohort)}
        
        # Add retention rates for each period
        for period in retention_matrix_pct.columns[:6]:  # Focus on first 6 periods
            if period in retention_matrix_pct.columns:
                cohort_data[f'period_{period}_retention'] = retention_matrix_pct.loc[cohort, period]
        
        # Calculate retention performance score
        retention_score = calculate_retention_score(retention_matrix_pct.loc[cohort])
        cohort_data['retention_score'] = retention_score
        
        # Performance category
        if retention_score >= 80:
            cohort_data['performance_category'] = 'Excellent'
        elif retention_score >= 60:
            cohort_data['performance_category'] = 'Good'
        elif retention_score >= 40:
            cohort_data['performance_category'] = 'Average'
        else:
            cohort_data['performance_category'] = 'Needs Improvement'
        
        trends_data.append(cohort_data)
    
    trends_df = pd.DataFrame(trends_data)
    return trends_df


def calculate_retention_score(cohort_retention_series):
    """
    Calculates a composite retention score for a cohort.
    Weights early retention more heavily as it's most critical.
    
    Returns:
        - float: Composite retention score (0-100)
    """
    
    if len(cohort_retention_series) < 2:
        return 0
    
    # Weight early periods more heavily
    weights = [0.4, 0.3, 0.2, 0.1]  # Weights for periods 1-4
    
    score = 0
    total_weight = 0
    
    for i, weight in enumerate(weights):
        if i + 1 < len(cohort_retention_series):  # i+1 because period 0 is acquisition
            period_retention = cohort_retention_series.iloc[i + 1]
            score += period_retention * weight
            total_weight += weight
    
    # Normalize score if we don't have all periods
    if total_weight > 0:
        score = score / total_weight
    
    return round(score, 2)


def cohort_retention_insights(retention_matrix_pct, cohort_sizes, retention_metrics):
    """
    Generates actionable business insights from cohort retention analysis.
    Business question: What strategic actions should we take based on retention patterns?
    Analysis:
        - Identifies successful retention strategies from high-performing cohorts.
        - Highlights areas needing improvement from low-performing cohorts.
        - Provides data-driven recommendations for customer retention programs.
    
    Returns:
        - dict: Strategic insights and actionable recommendations.
    """
    
    insights = {
        'performance_summary': '',
        'key_strengths': [],
        'improvement_areas': [],
        'strategic_recommendments': [],
        'retention_benchmarks': {}
    }
    
    # Performance summary
    avg_first_period_retention = retention_metrics.get('first_period_retention_avg', 0)
    
    if avg_first_period_retention >= 70:
        insights['performance_summary'] = 'Strong customer retention with excellent first-period engagement'
    elif avg_first_period_retention >= 50:
        insights['performance_summary'] = 'Moderate retention performance with room for improvement'
    else:
        insights['performance_summary'] = 'Critical need for retention improvement strategies'
    
    # Identify key strengths
    if retention_metrics.get('best_cohort_retention', 0) >= 80:
        insights['key_strengths'].append(
            f"Best cohort achieved {retention_metrics['best_cohort_retention']}% retention - analyze successful strategies"
        )
    
    if retention_metrics.get('cohort_size_growth_rate', 0) > 0:
        insights['key_strengths'].append(
            f"Positive cohort growth ({retention_metrics['cohort_size_growth_rate']}%) indicating acquisition effectiveness"
        )
    
    # Identify improvement areas
    if retention_metrics.get('retention_variability', 0) > 30:
        insights['improvement_areas'].append(
            f"High retention variability ({retention_metrics['retention_variability']}%) indicates inconsistent customer experience"
        )
    
    if retention_metrics.get('first_period_retention_avg', 0) < 40:
        insights['improvement_areas'].append(
            "Low first-period retention suggests issues with onboarding or initial product experience"
        )
    
    # Strategic recommendations
    if avg_first_period_retention < 50:
        insights['strategic_recommendments'].extend([
            "Implement enhanced onboarding program for new customers",
            "Develop welcome series with educational content",
            "Create early-engagement incentives for second purchase"
        ])
    
    if retention_metrics.get('long_term_retention_avg', 0) < 30:
        insights['strategic_recommendments'].extend([
            "Launch loyalty program with tiered benefits",
            "Develop personalized re-engagement campaigns",
            "Create exclusive content for long-term customers"
        ])
    
    # Retention benchmarks
    insights['retention_benchmarks'] = {
        'excellent_first_period': '> 70%',
        'good_first_period': '50-70%', 
        'needs_improvement': '< 50%',
        'industry_standard_ecommerce': '25-40% (varies by segment)'
    }
    
    return insights





