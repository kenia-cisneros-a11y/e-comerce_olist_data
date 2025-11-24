import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Tuple

class ProxyIndexCalculator:
    """
    Calculator for proxy return index based on 4 key metrics:
    1. On-time delivery
    2. Customer satisfaction (review score)
    3. Credit affinity/liquidity
    4. Healthy ticket size (potential)
    """
    
    def __init__(self, 
                 orders_df: pd.DataFrame,
                 order_items_df: pd.DataFrame,
                 order_payments_df: pd.DataFrame,
                 order_reviews_df: pd.DataFrame,
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize the proxy index calculator with required dataframes.
        
        Args:
            orders_df: Orders dataframe with delivery dates
            order_items_df: Order items dataframe with prices
            order_payments_df: Payments dataframe with payment types and installments
            order_reviews_df: Reviews dataframe with review scores
            weights: Optional weights for each metric (default: equal weights)
        """
        self.orders_df = orders_df
        self.order_items_df = order_items_df
        self.order_payments_df = order_payments_df
        self.order_reviews_df = order_reviews_df
        
        # Set default equal weights if not provided
        self.weights = weights or {
            'w1_delivery': 0.25,
            'w2_satisfaction': 0.25,
            'w3_credit': 0.25,
            'w4_ticket': 0.25
        }
    
    def calculate_on_time_delivery_score(self, order_id: str) -> float:
        """
        Calculate s1: On-time delivery score.
        s1 = 1 if order_delivered_customer_date <= order_estimated_delivery_date, else 0
        
        Args:
            order_id: Order identifier
            
        Returns:
            Score 1 if delivered on time, 0 otherwise
        """
        order = self.orders_df[self.orders_df['order_id'] == order_id]
        
        if order.empty:
            return 0.0
        
        delivered_date = pd.to_datetime(order['order_delivered_customer_date'].iloc[0])
        estimated_date = pd.to_datetime(order['order_estimated_delivery_date'].iloc[0])
        
        # Check if dates are valid
        if pd.isna(delivered_date) or pd.isna(estimated_date):
            return 0.0
        
        # Return 1 if delivered on time or early, 0 otherwise
        return 1.0 if delivered_date <= estimated_date else 0.0
    
    def calculate_satisfaction_score(self, order_id: str, cohort_avg: Optional[float] = None) -> float:
        """
        Calculate s2: Customer satisfaction score.
        s2 = review_score / 5 (if no review, use cohort average)
        
        Args:
            order_id: Order identifier
            cohort_avg: Average score for the cohort if no review exists
            
        Returns:
            Normalized satisfaction score (0-1)
        """
        review = self.order_reviews_df[self.order_reviews_df['order_id'] == order_id]
        
        if not review.empty and not pd.isna(review['review_score'].iloc[0]):
            # Use actual review score normalized to 0-1
            return review['review_score'].iloc[0] / 5.0
        elif cohort_avg is not None:
            # Use cohort average if no review exists
            return cohort_avg / 5.0
        else:
            # Calculate overall average as fallback
            avg_score = self.order_reviews_df['review_score'].mean()
            return avg_score / 5.0 if not pd.isna(avg_score) else 0.5
    
    def calculate_credit_affinity_score(self, order_id: str) -> float:
        """
        Calculate s3: Credit liquidity/affinity score.
        s3 = min((payment_installments / 12) + 1) if payment_type = 'CREDIT CARD'
        
        Args:
            order_id: Order identifier
            
        Returns:
            Credit affinity score
        """
        payments = self.order_payments_df[self.order_payments_df['order_id'] == order_id]
        
        if payments.empty:
            return 0.0
        
        # Filter for credit card payments
        credit_payments = payments[payments['payment_type'] == 'CREDIT CARD']
        
        if credit_payments.empty:
            return 0.0
        
        # Calculate score based on installments
        # Using max installments if multiple credit card payments
        max_installments = credit_payments['payment_installments'].max()
        
        # Apply formula: min((installments/12) + 1)
        # The min function here caps the value (though not specified in formula, assuming cap at 2)
        score = min((max_installments / 12.0) + 1.0, 2.0)
        
        return score
    
    def calculate_healthy_ticket_score(self, order_id: str, 
                                      p10: Optional[float] = None, 
                                      p90: Optional[float] = None) -> float:
        """
        Calculate s4: Healthy ticket score (potential).
        s4 = (AOV_order - P10) / (P90 - P10), capped at [0,1]
        
        Args:
            order_id: Order identifier
            p10: 10th percentile of order values (calculated if not provided)
            p90: 90th percentile of order values (calculated if not provided)
            
        Returns:
            Healthy ticket score (0-1)
        """
        # Calculate order value (AOV - Average Order Value for this order)
        order_items = self.order_items_df[self.order_items_df['order_id'] == order_id]
        
        if order_items.empty:
            return 0.0
        
        # Calculate total order value including freight
        order_value = order_items['price'].sum() + order_items['freight_value'].sum()
        
        # Calculate percentiles if not provided
        if p10 is None or p90 is None:
            # Calculate all order values for percentile calculation
            all_order_values = self.order_items_df.groupby('order_id').agg({
                'price': 'sum',
                'freight_value': 'sum'
            }).sum(axis=1)
            
            p10 = all_order_values.quantile(0.10) if p10 is None else p10
            p90 = all_order_values.quantile(0.90) if p90 is None else p90
        
        # Avoid division by zero
        if p90 - p10 == 0:
            return 0.5
        
        # Calculate score and cap at [0, 1]
        score = (order_value - p10) / (p90 - p10)
        return np.clip(score, 0.0, 1.0)
    
    def calculate_proxy_index(self, order_id: str, 
                            cohort_avg: Optional[float] = None,
                            p10: Optional[float] = None,
                            p90: Optional[float] = None) -> Tuple[float, Dict[str, float]]:
        """
        Calculate the complete proxy return index.
        i_retorno = w1*s1 + w2*s2 + w3*s3 + w4*s4
        
        Args:
            order_id: Order identifier
            cohort_avg: Average review score for cohort
            p10: 10th percentile of order values
            p90: 90th percentile of order values
            
        Returns:
            Tuple of (proxy_index, dict of individual scores)
        """
        # Calculate individual scores
        s1_delivery = self.calculate_on_time_delivery_score(order_id)
        s2_satisfaction = self.calculate_satisfaction_score(order_id, cohort_avg)
        s3_credit = self.calculate_credit_affinity_score(order_id)
        s4_ticket = self.calculate_healthy_ticket_score(order_id, p10, p90)
        
        # Calculate weighted index
        proxy_index = (
            self.weights['w1_delivery'] * s1_delivery +
            self.weights['w2_satisfaction'] * s2_satisfaction +
            self.weights['w3_credit'] * s3_credit +
            self.weights['w4_ticket'] * s4_ticket
        )
        
        # Return index and individual scores for transparency
        scores = {
            's1_delivery': s1_delivery,
            's2_satisfaction': s2_satisfaction,
            's3_credit': s3_credit,
            's4_ticket': s4_ticket
        }
        
        return proxy_index, scores
    
    def calculate_batch_proxy_indices(self, order_ids: Optional[list] = None) -> pd.DataFrame:
        """
        Calculate proxy indices for multiple orders.
        
        Args:
            order_ids: List of order IDs to process (if None, process all orders)
            
        Returns:
            DataFrame with order_id, proxy_index, and individual scores
        """
        if order_ids is None:
            order_ids = self.orders_df['order_id'].unique()
        
        # Pre-calculate global statistics for efficiency
        cohort_avg = self.order_reviews_df['review_score'].mean()
        
        all_order_values = self.order_items_df.groupby('order_id').agg({
            'price': 'sum',
            'freight_value': 'sum'
        }).sum(axis=1)
        
        p10 = all_order_values.quantile(0.10)
        p90 = all_order_values.quantile(0.90)
        
        results = []
        for order_id in order_ids:
            proxy_index, scores = self.calculate_proxy_index(
                order_id, cohort_avg, p10, p90
            )
            
            results.append({
                'order_id': order_id,
                'proxy_index': proxy_index,
                **scores
            })
        
        return pd.DataFrame(results)


def main():
    """
    Example usage of the ProxyIndexCalculator
    """
    # Example: Load your dataframes
    # orders_df = pd.read_csv('orders.csv')
    # order_items_df = pd.read_csv('order_items.csv')
    # order_payments_df = pd.read_csv('order_payments.csv')
    # order_reviews_df = pd.read_csv('order_reviews.csv')
    
    # Example with dummy data
    orders_df = pd.DataFrame({
        'order_id': ['ORD001', 'ORD002'],
        'order_delivered_customer_date': ['2024-01-15', '2024-01-20'],
        'order_estimated_delivery_date': ['2024-01-16', '2024-01-18']
    })
    
    order_items_df = pd.DataFrame({
        'order_id': ['ORD001', 'ORD001', 'ORD002'],
        'price': [100.0, 50.0, 200.0],
        'freight_value': [10.0, 5.0, 15.0]
    })
    
    order_payments_df = pd.DataFrame({
        'order_id': ['ORD001', 'ORD002'],
        'payment_type': ['CREDIT CARD', 'boleto'],
        'payment_installments': [3, 1]
    })
    
    order_reviews_df = pd.DataFrame({
        'order_id': ['ORD001', 'ORD002'],
        'review_score': [4, 5]
    })
    
    # Initialize calculator with custom weights
    weights = {
        'w1_delivery': 0.30,
        'w2_satisfaction': 0.25,
        'w3_credit': 0.20,
        'w4_ticket': 0.25
    }
    
    calculator = ProxyIndexCalculator(
        orders_df=orders_df,
        order_items_df=order_items_df,
        order_payments_df=order_payments_df,
        order_reviews_df=order_reviews_df,
        weights=weights
    )
    
    # Calculate for all orders
    results_df = calculator.calculate_batch_proxy_indices()
    
    print("Proxy Index Calculation Results:")
    print(results_df.to_string(index=False))
    print("\nSummary Statistics:")
    print(results_df[['proxy_index', 's1_delivery', 's2_satisfaction', 's3_credit', 's4_ticket']].describe())


if __name__ == "__main__":
    main()
