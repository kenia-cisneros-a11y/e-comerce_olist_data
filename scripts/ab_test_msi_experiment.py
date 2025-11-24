"""
A/B Test Framework for MSI (Meses Sin Intereses) Experiment
Objective: Evaluate if offering more months without interest increases 60-day return intention
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import hashlib
import json
from proxy_index_calculator import ProxyIndexCalculator

class MSIExperiment:
    """
    A/B Test framework for evaluating the impact of extended MSI on return intention.
    
    H0: Proxy return index is equal between Treatment and Control
    H1: Proxy return index is higher in Treatment (one-tailed test)
    """
    
    def __init__(self, 
                 start_date: str,
                 end_date: str,
                 db_connection=None,
                 config_path: Optional[str] = None):
        """
        Initialize the MSI experiment.
        
        Args:
            start_date: Experiment enrollment start date
            end_date: Experiment enrollment end date
            db_connection: Database connection
            config_path: Path to experiment configuration file
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.observation_window_days = 60
        self.db_connection = db_connection
        
        # Load or set default configuration
        self.config = self._load_config(config_path) if config_path else self._default_config()
        
        # Initialize results storage
        self.enrollment_data = None
        self.results = None
        self.guardrails = None
        
    def _default_config(self) -> Dict:
        """
        Default experiment configuration.
        """
        return {
            "experiment_name": "MSI_Extension_Impact",
            "treatment": {
                "msi_months": [12, 18, 24],  # Extended MSI options
                "description": "Extended months without interest"
            },
            "control": {
                "msi_months": [3, 6],  # Standard MSI options
                "description": "Standard months without interest"
            },
            "stratification": {
                "category": True,
                "state": True,
                "aov_percentile": True
            },
            "allocation": {
                "treatment_ratio": 0.5,
                "seed": 42
            },
            "metrics": {
                "primary": "proxy_index",
                "secondary": ["s1_delivery", "s2_satisfaction", "payment_installments", "aov"],
                "guardrails": ["payment_approval_rate", "cancellation_rate", 
                             "delivery_time", "raw_review_score", "order_margin"]
            },
            "analysis": {
                "confidence_level": 0.95,
                "test_type": "one_tailed",
                "min_sample_size": 1000
            }
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load experiment configuration from JSON file.
        """
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def assign_treatment(self, customer_id: str, 
                        category: str, 
                        state: str, 
                        aov: float) -> str:
        """
        Assign customer to treatment or control group using stratified randomization.
        
        Args:
            customer_id: Unique customer identifier
            category: Product category for stratification
            state: Customer state for stratification
            aov: Order value for percentile stratification
            
        Returns:
            'treatment' or 'control'
        """
        # Create stratification key
        strata_components = []
        
        if self.config['stratification']['product_category_name']:
            strata_components.append(f"cat_{category}")
        
        if self.config['stratification']['state']:
            strata_components.append(f"state_{state}")
        
        if self.config['stratification']['aov_percentile']:
            # Calculate AOV percentile bucket (quartiles)
            if aov < 100:
                aov_bucket = "Q1"
            elif aov < 250:
                aov_bucket = "Q2"
            elif aov < 500:
                aov_bucket = "Q3"
            else:
                aov_bucket = "Q4"
            strata_components.append(f"aov_{aov_bucket}")
        
        strata_key = "_".join(strata_components)
        
        # Hash-based assignment for consistency
        hash_input = f"{customer_id}_{strata_key}_{self.config['allocation']['seed']}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Assign based on ratio
        threshold = self.config['allocation']['treatment_ratio']
        assignment_value = (hash_value % 100) / 100
        
        return 'treatment' if assignment_value < threshold else 'control'
    
    def enroll_customers(self, orders_df: pd.DataFrame, 
                        order_items_df: pd.DataFrame,
                        order_payments_df: pd.DataFrame,
                        products_df: pd.DataFrame,
                        customers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enroll eligible customers in the experiment.
        
        Args:
            orders_df: Orders data with payment information
            order_items_df: Order items for category extraction
            customers_df: Customer data with location
            
        Returns:
            DataFrame with enrolled customers and assignments
        """
        # Filter orders within enrollment window
        enrollment_orders = orders_df[
            (pd.to_datetime(orders_df['order_purchase_timestamp']) >= self.start_date) &
            (pd.to_datetime(orders_df['order_purchase_timestamp']) <= self.end_date)
        ].copy()

        # to add 'payment_type'
        enrollment_orders = orders_df.merge(
            order_payments_df,
            on='order_id',
            how='left'
        )
        
        # Filter for credit card payments only (eligible population)
        credit_orders = enrollment_orders[
            enrollment_orders['payment_type'] == 'CREDIT CARD'
        ].copy()
        
        # Join with order items to get category
        credit_orders = credit_orders.merge(
            order_items_df.groupby('order_id').agg({
                'product_id': 'first',  # Primary product
                'price': 'sum',
                'freight_value': 'sum'
            }).reset_index(),
            on='order_id',
            how='left'
        ).merge(
            products_df[['product_id', 'product_category_name']],
            on='product_id',
            how='left'
        )
        
        # Calculate AOV
        credit_orders['aov'] = credit_orders['price'] + credit_orders['freight_value']
        
        # Join with customer data
        credit_orders = credit_orders.merge(
            customers_df[['customer_id', 'customer_state']],
            on='customer_id',
            how='left'
        )

        # print(credit_orders.columns.tolist())
        # print([c for c in credit_orders.columns if c == 'product_category_name'])
        
        # Assign treatment
        credit_orders['assignment'] = credit_orders.apply(
            lambda row: self.assign_treatment(
                row['customer_id'],
                row['product_category_name'],
                row['customer_state'],
                row['aov']
            ),
            axis=1
        )
        
        # Add experiment metadata
        credit_orders['enrollment_date'] = credit_orders['order_purchase_timestamp']
        credit_orders['observation_end_date'] = pd.to_datetime(
            credit_orders['enrollment_date']
        ) + timedelta(days=self.observation_window_days)
        
        # Store enrollment data
        self.enrollment_data = credit_orders
        
        # Log enrollment statistics
        print(f"Experiment Enrollment Summary:")
        print(f"Total enrolled: {len(credit_orders)}")
        print(f"Treatment: {len(credit_orders[credit_orders['assignment'] == 'treatment'])}")
        print(f"Control: {len(credit_orders[credit_orders['assignment'] == 'control'])}")
        print(f"Balance: {credit_orders['assignment'].value_counts(normalize=True).to_dict()}")
        
        return credit_orders
    
    def apply_treatment(self, order_id: str, assignment: str) -> Dict:
        """
        Apply treatment configuration to an order.
        
        Args:
            order_id: Order identifier
            assignment: 'treatment' or 'control'
            
        Returns:
            Dictionary with applied configuration
        """
        if assignment == 'treatment':
            msi_options = self.config['treatment']['msi_months']
            max_installments = max(msi_options)
            available_plans = msi_options
        else:
            msi_options = self.config['control']['msi_months']
            max_installments = max(msi_options)
            available_plans = msi_options
        
        return {
            'order_id': order_id,
            'max_installments': max_installments,
            'available_plans': available_plans,
            'msi_displayed': True
        }
    
    def calculate_metrics(self, 
                         enrolled_df: pd.DataFrame,
                         orders_df: pd.DataFrame,
                         order_items_df: pd.DataFrame,
                         order_payments_df: pd.DataFrame,
                         order_reviews_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate proxy index and component metrics for enrolled customers.
        
        Args:
            enrolled_df: Enrolled customers with assignments
            orders_df: Complete orders data
            order_items_df: Order items data
            order_payments_df: Payment data
            order_reviews_df: Reviews data
            
        Returns:
            DataFrame with calculated metrics
        """
        # Initialize proxy calculator
        calculator = ProxyIndexCalculator(
            orders_df=orders_df,
            order_items_df=order_items_df,
            order_payments_df=order_payments_df,
            order_reviews_df=order_reviews_df
        )
        
        results = []
        
        for _, row in enrolled_df.iterrows():
            order_id = row['order_id']
            
            # Calculate proxy index and components
            proxy_index, scores = calculator.calculate_proxy_index(order_id)
            
            # Get actual installments chosen
            payment_data = order_payments_df[
                order_payments_df['order_id'] == order_id
            ]
            installments = payment_data['payment_installments'].max() if not payment_data.empty else 0
            
            # Compile results
            result = {
                'order_id': order_id,
                'customer_id': row['customer_id'],
                'assignment': row['assignment'],
                'product_category_name': row['product_category_name'],
                'state': row['customer_state'],
                'aov': row['aov'],
                'enrollment_date': row['enrollment_date'],
                'proxy_index': proxy_index,
                's1_delivery': scores['s1_delivery'],
                's2_satisfaction': scores['s2_satisfaction'],
                's3_credit': scores['s3_credit'],
                's4_ticket': scores['s4_ticket'],
                'payment_installments': installments
            }
            
            results.append(result)
        
        results_df = pd.DataFrame(results)
        self.results = results_df
        
        return results_df
    
    def calculate_guardrails(self, 
                            enrolled_df: pd.DataFrame,
                            orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate guardrail metrics to ensure experiment safety.
        
        Args:
            enrolled_df: Enrolled customers
            orders_df: Complete orders data
            
        Returns:
            DataFrame with guardrail metrics by group
        """
        guardrails = []
        
        for group in ['treatment', 'control']:
            group_orders = enrolled_df[enrolled_df['assignment'] == group]
            
            # Payment approval rate (simplified - would need payment status)
            approval_rate = 0.95  # Placeholder
            
            # Cancellation rate
            cancellation_rate = len(
                group_orders[group_orders['order_status'] == 'canceled']
            ) / len(group_orders) if len(group_orders) > 0 else 0
            
            # Average delivery time
            delivery_times = pd.to_datetime(
                group_orders['order_delivered_customer_date']
            ) - pd.to_datetime(
                group_orders['order_purchase_timestamp']
            )
            avg_delivery_days = delivery_times.dt.days.mean()
            
            # Raw review score
            review_scores = enrolled_df[enrolled_df['assignment'] == group].merge(
                orders_df[['order_id', 'review_score']],
                on='order_id',
                how='left'
            )['review_score'].mean()
            
            # Order margin (simplified calculation)
            avg_margin = group_orders['aov'].mean() * 0.2  # Assuming 20% margin
            
            guardrails.append({
                'group': group,
                'payment_approval_rate': approval_rate,
                'cancellation_rate': cancellation_rate,
                'avg_delivery_days': avg_delivery_days,
                'avg_review_score': review_scores,
                'avg_order_margin': avg_margin
            })
        
        guardrails_df = pd.DataFrame(guardrails)
        self.guardrails = guardrails_df
        
        return guardrails_df
    
    def generate_experiment_report(self) -> Dict:
        """
        Generate comprehensive experiment report.
        
        Returns:
            Dictionary with experiment results and analysis
        """
        if self.results is None:
            raise ValueError("No results calculated yet. Run calculate_metrics first.")
        
        treatment_results = self.results[self.results['assignment'] == 'treatment']
        control_results = self.results[self.results['assignment'] == 'control']
        
        report = {
            'experiment_info': {
                'name': self.config['experiment_name'],
                'start_date': str(self.start_date),
                'end_date': str(self.end_date),
                'observation_window': self.observation_window_days
            },
            'sample_sizes': {
                'treatment': len(treatment_results),
                'control': len(control_results),
                'total': len(self.results)
            },
            'primary_metric': {
                'metric_name': 'proxy_index',
                'treatment_mean': treatment_results['proxy_index'].mean(),
                'control_mean': control_results['proxy_index'].mean(),
                'absolute_lift': treatment_results['proxy_index'].mean() - control_results['proxy_index'].mean(),
                'relative_lift': ((treatment_results['proxy_index'].mean() / 
                                 control_results['proxy_index'].mean()) - 1) * 100
            },
            'secondary_metrics': {
                's1_delivery': {
                    'treatment': treatment_results['s1_delivery'].mean(),
                    'control': control_results['s1_delivery'].mean()
                },
                's2_satisfaction': {
                    'treatment': treatment_results['s2_satisfaction'].mean(),
                    'control': control_results['s2_satisfaction'].mean()
                },
                'avg_installments': {
                    'treatment': treatment_results['payment_installments'].mean(),
                    'control': control_results['payment_installments'].mean()
                },
                'avg_order_value': {
                    'treatment': treatment_results['aov'].mean(),
                    'control': control_results['aov'].mean()
                }
            },
            'stratification_balance': {
                'product_category_name': self._check_balance('product_category_name'),
                'state': self._check_balance('state'),
                'aov_quartiles': self._check_balance_numeric('aov')
            }
        }
        
        if self.guardrails is not None:
            report['guardrails'] = self.guardrails.to_dict('records')
        
        return report
    
    def _check_balance(self, column: str) -> Dict:
        """
        Check balance of categorical variable between groups.
        """
        treatment_dist = self.results[
            self.results['assignment'] == 'treatment'
        ][column].value_counts(normalize=True).to_dict()
        
        control_dist = self.results[
            self.results['assignment'] == 'control'
        ][column].value_counts(normalize=True).to_dict()
        
        return {
            'treatment_distribution': treatment_dist,
            'control_distribution': control_dist
        }
    
    def _check_balance_numeric(self, column: str) -> Dict:
        """
        Check balance of numeric variable between groups.
        """
        treatment_stats = self.results[
            self.results['assignment'] == 'treatment'
        ][column].describe().to_dict()
        
        control_stats = self.results[
            self.results['assignment'] == 'control'
        ][column].describe().to_dict()
        
        return {
            'treatment_stats': treatment_stats,
            'control_stats': control_stats
        }
    
    def save_results(self, output_dir: str = '/mnt/user-data/outputs'):
        """
        Save experiment results to files.
        
        Args:
            output_dir: Directory to save results
        """
        # Save enrollment data
        if self.enrollment_data is not None:
            self.enrollment_data.to_csv(
                f'{output_dir}/msi_experiment_enrollment.csv', 
                index=False
            )
        
        # Save results
        if self.results is not None:
            self.results.to_csv(
                f'{output_dir}/msi_experiment_results.csv', 
                index=False
            )
        
        # Save guardrails
        if self.guardrails is not None:
            self.guardrails.to_csv(
                f'{output_dir}/msi_experiment_guardrails.csv', 
                index=False
            )
        
        # Save report
        report = self.generate_experiment_report()
        with open(f'{output_dir}/msi_experiment_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Results saved to {output_dir}")


def main():
    """
    Example usage of MSI Experiment
    """
    # Initialize experiment
    experiment = MSIExperiment(
        start_date='2024-01-01',
        end_date='2024-01-31'
    )
    
    print("MSI A/B Test Experiment initialized")
    print(f"Treatment: {experiment.config['treatment']['msi_months']} months")
    print(f"Control: {experiment.config['control']['msi_months']} months")
    

if __name__ == "__main__":
    main()
