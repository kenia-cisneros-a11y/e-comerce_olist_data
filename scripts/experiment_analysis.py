"""
Statistical Analysis Module for MSI Experiment
Implements hypothesis testing, regression models, and heterogeneity analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class ExperimentAnalysis:
    """
    Statistical analysis for A/B test experiments.
    Implements various statistical tests and models for causal inference.
    """
    
    def __init__(self, results_df: pd.DataFrame, 
                 confidence_level: float = 0.95,
                 test_type: str = 'one_tailed'):
        """
        Initialize analysis with experiment results.
        
        Args:
            results_df: DataFrame with experiment results
            confidence_level: Confidence level for tests (default 0.95)
            test_type: 'one_tailed' or 'two_tailed'
        """
        self.results = results_df
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.test_type = test_type
        
        # Separate treatment and control groups
        self.treatment = results_df[results_df['assignment'] == 'treatment']
        self.control = results_df[results_df['assignment'] == 'control']
        
        # Store analysis results
        self.primary_test_results = None
        self.regression_results = None
        self.heterogeneity_results = None
    
    def perform_primary_analysis(self) -> Dict:
        """
        Perform primary hypothesis test on proxy index.
        H0: μ_treatment = μ_control
        H1: μ_treatment > μ_control (one-tailed)
        
        Returns:
            Dictionary with test results
        """
        # Extract proxy indices
        treatment_indices = self.treatment['proxy_index'].dropna()
        control_indices = self.control['proxy_index'].dropna()
        
        # Calculate descriptive statistics
        treatment_stats = {
            'mean': treatment_indices.mean(),
            'std': treatment_indices.std(),
            'sem': treatment_indices.sem(),
            'n': len(treatment_indices)
        }
        
        control_stats = {
            'mean': control_indices.mean(),
            'std': control_indices.std(),
            'sem': control_indices.sem(),
            'n': len(control_indices)
        }
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((treatment_stats['n'] - 1) * treatment_stats['std']**2 + 
             (control_stats['n'] - 1) * control_stats['std']**2) /
            (treatment_stats['n'] + control_stats['n'] - 2)
        )
        cohens_d = (treatment_stats['mean'] - control_stats['mean']) / pooled_std
        
        # Perform t-test
        if self.test_type == 'one_tailed':
            t_stat, p_value_two_tailed = ttest_ind(
                treatment_indices, 
                control_indices,
                equal_var=True
            )
            p_value = p_value_two_tailed / 2 if t_stat > 0 else 1 - (p_value_two_tailed / 2)
        else:
            t_stat, p_value = ttest_ind(
                treatment_indices, 
                control_indices,
                equal_var=True
            )
        
        # Calculate confidence interval for difference
        diff = treatment_stats['mean'] - control_stats['mean']
        se_diff = np.sqrt(
            treatment_stats['sem']**2 + control_stats['sem']**2
        )
        
        if self.test_type == 'one_tailed':
            # One-sided CI
            t_critical = stats.t.ppf(self.confidence_level, 
                                    treatment_stats['n'] + control_stats['n'] - 2)
            ci_lower = diff - t_critical * se_diff
            ci_upper = np.inf
        else:
            # Two-sided CI
            t_critical = stats.t.ppf(1 - self.alpha/2, 
                                    treatment_stats['n'] + control_stats['n'] - 2)
            ci_lower = diff - t_critical * se_diff
            ci_upper = diff + t_critical * se_diff
        
        # Perform non-parametric test (Mann-Whitney U) as robustness check
        if self.test_type == 'one_tailed':
            u_stat, p_value_mw = mannwhitneyu(
                treatment_indices,
                control_indices,
                alternative='greater'
            )
        else:
            u_stat, p_value_mw = mannwhitneyu(
                treatment_indices,
                control_indices,
                alternative='two-sided'
            )
        
        # Test for normality (for validation)
        _, p_norm_treatment = stats.shapiro(treatment_indices[:min(5000, len(treatment_indices))])
        _, p_norm_control = stats.shapiro(control_indices[:min(5000, len(control_indices))])
        
        # Calculate statistical power (post-hoc)
        from statsmodels.stats.power import ttest_power
        observed_power = ttest_power(
            cohens_d,
            treatment_stats['n'],
            self.alpha,
            alternative='larger' if self.test_type == 'one_tailed' else 'two-sided'
        )
        
        results = {
            'hypothesis_test': {
                'null_hypothesis': 'μ_treatment = μ_control',
                'alternative_hypothesis': 'μ_treatment > μ_control' if self.test_type == 'one_tailed' else 'μ_treatment ≠ μ_control',
                'test_type': self.test_type,
                'alpha': self.alpha
            },
            'descriptive_stats': {
                'treatment': treatment_stats,
                'control': control_stats
            },
            'effect_size': {
                'absolute_difference': diff,
                'relative_lift': (diff / control_stats['mean']) * 100,
                'cohens_d': cohens_d
            },
            't_test': {
                't_statistic': t_stat,
                'p_value': p_value,
                'degrees_of_freedom': treatment_stats['n'] + control_stats['n'] - 2,
                'reject_null': p_value < self.alpha,
                'confidence_interval': {
                    'lower': ci_lower,
                    'upper': ci_upper
                }
            },
            'mann_whitney_test': {
                'u_statistic': u_stat,
                'p_value': p_value_mw,
                'reject_null': p_value_mw < self.alpha
            },
            'assumptions': {
                'normality_treatment': {
                    'p_value': p_norm_treatment,
                    'is_normal': p_norm_treatment > 0.05
                },
                'normality_control': {
                    'p_value': p_norm_control,
                    'is_normal': p_norm_control > 0.05
                }
            },
            'power_analysis': {
                'observed_power': observed_power,
                'is_adequate': observed_power > 0.8
            }
        }
        
        self.primary_test_results = results
        return results
    
    def perform_regression_analysis(self) -> Dict:
        """
        Perform adjusted regression analysis with covariates.
        Model: proxy_index ~ treatment + category + state + aov
        
        Returns:
            Dictionary with regression results
        """
        # Prepare data for regression
        regression_data = self.results.copy()
        
        # Create treatment indicator
        regression_data['treatment_indicator'] = (
            regression_data['assignment'] == 'treatment'
        ).astype(int)
        
        # Encode categorical variables
        le_category = LabelEncoder()
        le_state = LabelEncoder()
        
        regression_data['category_encoded'] = le_category.fit_transform(
            regression_data['product_category_name'].fillna('unknown')
        )
        regression_data['state_encoded'] = le_state.fit_transform(
            regression_data['state'].fillna('unknown')
        )
        
        # Standardize AOV
        scaler = StandardScaler()
        regression_data['aov_standardized'] = scaler.fit_transform(
            regression_data[['aov']]
        )
        
        # Linear regression for proxy index
        formula = 'proxy_index ~ treatment_indicator + C(category) + C(state) + aov_standardized'
        
        try:
            model = smf.ols(formula, data=regression_data).fit()
            
            # Get robust standard errors (HC3)
            robust_model = model.get_robustcov_results(cov_type='HC3')
            
            linear_results = {
                'formula': formula,
                'coefficients': {
                    'treatment_effect': robust_model.params['treatment_indicator'],
                    'treatment_se': robust_model.bse['treatment_indicator'],
                    'treatment_pvalue': robust_model.pvalues['treatment_indicator']
                },
                'model_fit': {
                    'r_squared': model.rsquared,
                    'adj_r_squared': model.rsquared_adj,
                    'f_statistic': model.fvalue,
                    'f_pvalue': model.f_pvalue,
                    'aic': model.aic,
                    'bic': model.bic
                },
                'full_summary': str(robust_model.summary())
            }
        except Exception as e:
            linear_results = {'error': str(e)}
        
        # Logistic regression for s1 (delivery)
        try:
            X = regression_data[['treatment_indicator', 'category_encoded', 
                                'state_encoded', 'aov_standardized']].values
            y = regression_data['s1_delivery'].values
            
            log_model_s1 = LogisticRegression(max_iter=1000)
            log_model_s1.fit(X, y)
            
            # Calculate odds ratio for treatment effect
            treatment_coef = log_model_s1.coef_[0][0]
            odds_ratio = np.exp(treatment_coef)
            
            logistic_s1_results = {
                'outcome': 's1_delivery',
                'treatment_coefficient': treatment_coef,
                'odds_ratio': odds_ratio,
                'interpretation': f"Treatment {'increases' if odds_ratio > 1 else 'decreases'} odds of on-time delivery by {abs(odds_ratio-1)*100:.1f}%"
            }
        except Exception as e:
            logistic_s1_results = {'error': str(e)}
        
        # Ordinal regression for s2 (satisfaction)
        try:
            # Simplified: treating as continuous for now
            formula_s2 = 's2_satisfaction ~ treatment_indicator + C(category) + C(state) + aov_standardized'
            model_s2 = smf.ols(formula_s2, data=regression_data).fit()
            
            ordinal_s2_results = {
                'outcome': 's2_satisfaction',
                'treatment_effect': model_s2.params['treatment_indicator'],
                'treatment_pvalue': model_s2.pvalues['treatment_indicator'],
                'interpretation': f"Treatment {'increases' if model_s2.params['treatment_indicator'] > 0 else 'decreases'} satisfaction score by {abs(model_s2.params['treatment_indicator']):.3f}"
            }
        except Exception as e:
            ordinal_s2_results = {'error': str(e)}
        
        results = {
            'linear_regression': linear_results,
            'logistic_regression_s1': logistic_s1_results,
            'ordinal_regression_s2': ordinal_s2_results
        }
        
        self.regression_results = results
        return results
    
    def perform_heterogeneity_analysis(self) -> Dict:
        """
        Analyze heterogeneous treatment effects across subgroups.
        
        Returns:
            Dictionary with heterogeneity analysis results
        """
        heterogeneity_results = {}
        
        # Analyze by category
        category_effects = []
        for category in self.results['product_category_name'].unique():
            category_data = self.results[self.results['product_category_name'] == category]
            
            treatment_cat = category_data[category_data['assignment'] == 'treatment']['proxy_index']
            control_cat = category_data[category_data['assignment'] == 'control']['proxy_index']
            
            if len(treatment_cat) > 0 and len(control_cat) > 0:
                effect = treatment_cat.mean() - control_cat.mean()
                t_stat, p_value = ttest_ind(treatment_cat, control_cat)
                
                category_effects.append({
                    'category': category,
                    'treatment_mean': treatment_cat.mean(),
                    'control_mean': control_cat.mean(),
                    'effect': effect,
                    'relative_lift': (effect / control_cat.mean() * 100) if control_cat.mean() > 0 else 0,
                    'sample_size': len(category_data),
                    'p_value': p_value,
                    'significant': p_value < self.alpha
                })
        
        heterogeneity_results['by_category'] = category_effects
        
        # Analyze by state (top 5 states by sample size)
        top_states = self.results['state'].value_counts().head(5).index
        state_effects = []
        
        for state in top_states:
            state_data = self.results[self.results['state'] == state]
            
            treatment_state = state_data[state_data['assignment'] == 'treatment']['proxy_index']
            control_state = state_data[state_data['assignment'] == 'control']['proxy_index']
            
            if len(treatment_state) > 0 and len(control_state) > 0:
                effect = treatment_state.mean() - control_state.mean()
                t_stat, p_value = ttest_ind(treatment_state, control_state)
                
                state_effects.append({
                    'state': state,
                    'treatment_mean': treatment_state.mean(),
                    'control_mean': control_state.mean(),
                    'effect': effect,
                    'relative_lift': (effect / control_state.mean() * 100) if control_state.mean() > 0 else 0,
                    'sample_size': len(state_data),
                    'p_value': p_value,
                    'significant': p_value < self.alpha
                })
        
        heterogeneity_results['by_state'] = state_effects
        
        # Analyze by AOV quartile
        aov_quartiles = pd.qcut(self.results['aov'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        self.results['aov_quartile'] = aov_quartiles
        
        aov_effects = []
        for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
            quartile_data = self.results[self.results['aov_quartile'] == quartile]
            
            treatment_q = quartile_data[quartile_data['assignment'] == 'treatment']['proxy_index']
            control_q = quartile_data[quartile_data['assignment'] == 'control']['proxy_index']
            
            if len(treatment_q) > 0 and len(control_q) > 0:
                effect = treatment_q.mean() - control_q.mean()
                t_stat, p_value = ttest_ind(treatment_q, control_q)
                
                aov_effects.append({
                    'quartile': quartile,
                    'treatment_mean': treatment_q.mean(),
                    'control_mean': control_q.mean(),
                    'effect': effect,
                    'relative_lift': (effect / control_q.mean() * 100) if control_q.mean() > 0 else 0,
                    'sample_size': len(quartile_data),
                    'p_value': p_value,
                    'significant': p_value < self.alpha
                })
        
        heterogeneity_results['by_aov_quartile'] = aov_effects
        
        # Test for interaction effects
        try:
            # Model with interaction terms
            interaction_formula = '''proxy_index ~ treatment_indicator * C(category) + 
                                                 treatment_indicator * C(state) + 
                                                 treatment_indicator * aov_standardized'''
            
            regression_data = self.results.copy()
            regression_data['treatment_indicator'] = (
                regression_data['assignment'] == 'treatment'
            ).astype(int)
            
            scaler = StandardScaler()
            regression_data['aov_standardized'] = scaler.fit_transform(
                regression_data[['aov']]
            )
            
            interaction_model = smf.ols(interaction_formula, data=regression_data).fit()
            
            # Extract interaction p-values
            interaction_pvalues = {
                k: v for k, v in interaction_model.pvalues.items() 
                if 'treatment_indicator:' in k
            }
            
            significant_interactions = [
                k for k, v in interaction_pvalues.items() 
                if v < self.alpha
            ]
            
            heterogeneity_results['interaction_effects'] = {
                'significant_interactions': significant_interactions,
                'all_interaction_pvalues': interaction_pvalues
            }
        except Exception as e:
            heterogeneity_results['interaction_effects'] = {'error': str(e)}
        
        self.heterogeneity_results = heterogeneity_results
        return heterogeneity_results
    
    def analyze_secondary_metrics(self) -> Dict:
        """
        Analyze secondary metrics (s1, s2, installments, AOV).
        
        Returns:
            Dictionary with secondary metric analysis
        """
        secondary_results = {}
        
        metrics = ['s1_delivery', 's2_satisfaction', 'payment_installments', 'aov']
        
        for metric in metrics:
            treatment_values = self.treatment[metric].dropna()
            control_values = self.control[metric].dropna()
            
            # Perform appropriate test based on metric type
            if metric == 's1_delivery':
                # Binary metric - use chi-square test
                treatment_rate = treatment_values.mean()
                control_rate = control_values.mean()
                
                # Create contingency table
                treatment_success = int(treatment_values.sum())
                treatment_fail = len(treatment_values) - treatment_success
                control_success = int(control_values.sum())
                control_fail = len(control_values) - control_success
                
                contingency_table = np.array([
                    [treatment_success, treatment_fail],
                    [control_success, control_fail]
                ])
                
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                
                secondary_results[metric] = {
                    'treatment_rate': treatment_rate,
                    'control_rate': control_rate,
                    'absolute_difference': treatment_rate - control_rate,
                    'relative_lift': ((treatment_rate / control_rate) - 1) * 100 if control_rate > 0 else 0,
                    'chi2_statistic': chi2,
                    'p_value': p_value,
                    'significant': p_value < self.alpha
                }
            else:
                # Continuous metric - use t-test
                t_stat, p_value = ttest_ind(treatment_values, control_values)
                
                secondary_results[metric] = {
                    'treatment_mean': treatment_values.mean(),
                    'treatment_std': treatment_values.std(),
                    'control_mean': control_values.mean(),
                    'control_std': control_values.std(),
                    'absolute_difference': treatment_values.mean() - control_values.mean(),
                    'relative_lift': ((treatment_values.mean() / control_values.mean()) - 1) * 100 if control_values.mean() > 0 else 0,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < self.alpha
                }
        
        # Analyze installment distribution
        treatment_installments = self.treatment['payment_installments'].value_counts().sort_index()
        control_installments = self.control['payment_installments'].value_counts().sort_index()
        
        secondary_results['installment_distribution'] = {
            'treatment': treatment_installments.to_dict(),
            'control': control_installments.to_dict()
        }
        
        return secondary_results
    
    def generate_full_report(self) -> Dict:
        """
        Generate comprehensive statistical analysis report.
        
        Returns:
            Dictionary with all analysis results
        """
        # Run all analyses
        primary = self.perform_primary_analysis()
        regression = self.perform_regression_analysis()
        heterogeneity = self.perform_heterogeneity_analysis()
        secondary = self.analyze_secondary_metrics()
        
        # Compile recommendations
        recommendations = self._generate_recommendations()
        
        report = {
            'analysis_summary': {
                'conclusion': self._get_conclusion(),
                'confidence_level': self.confidence_level,
                'test_type': self.test_type,
                'sample_size': len(self.results)
            },
            'primary_analysis': primary,
            'regression_analysis': regression,
            'heterogeneity_analysis': heterogeneity,
            'secondary_metrics': secondary,
            'recommendations': recommendations
        }
        
        return report
    
    def _get_conclusion(self) -> str:
        """
        Generate conclusion based on primary test results.
        """
        if self.primary_test_results is None:
            return "Analysis not yet performed"
        
        if self.primary_test_results['t_test']['reject_null']:
            effect = self.primary_test_results['effect_size']['relative_lift']
            return f"POSITIVE: Extended MSI significantly increases return intention by {effect:.1f}% (p < {self.alpha})"
        else:
            return f"NEUTRAL: No significant difference in return intention between extended and standard MSI (p = {self.primary_test_results['t_test']['p_value']:.3f})"
    
    def _generate_recommendations(self) -> List[str]:
        """
        Generate actionable recommendations based on analysis.
        """
        recommendations = []
        
        if self.primary_test_results and self.primary_test_results['t_test']['reject_null']:
            recommendations.append("Consider rolling out extended MSI options to all customers")
            
            if self.heterogeneity_results:
                # Check for significant heterogeneity
                sig_categories = [
                    c['product_category_name'] for c in self.heterogeneity_results['by_category'] 
                    if c['significant'] and c['effect'] > 0
                ]
                if sig_categories:
                    recommendations.append(
                        f"Prioritize extended MSI for categories: {', '.join(sig_categories)}"
                    )
        else:
            recommendations.append("Extended MSI does not show significant impact - maintain current offering")
            recommendations.append("Consider testing other interventions to increase return intention")
        
        # Power analysis recommendation
        if self.primary_test_results and not self.primary_test_results['power_analysis']['is_adequate']:
            recommendations.append("Increase sample size for future experiments to achieve 80% statistical power")
        
        return recommendations


def main():
    """
    Example usage of ExperimentAnalysis
    """
    print("Statistical Analysis Module for MSI Experiment")
    print("=" * 60)
    print("Available analyses:")
    print("1. Primary hypothesis testing (t-test, Mann-Whitney)")
    print("2. Regression analysis with covariates")
    print("3. Heterogeneity analysis by subgroups")
    print("4. Secondary metrics analysis")
    print("5. Full statistical report generation")


if __name__ == "__main__":
    main()
