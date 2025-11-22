import pandas as pd
import sqlite3
import numpy as np
# from ydata_profiling import ProfileReport

import warnings

# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)


db_path = 'C:/Users/kenia/OneDrive/Documents/GitHub/cobre/data/olist.sqlite'
db_connection = sqlite3.connect(db_path, check_same_thread = False)

def load_table(table):
    query = f"""
        SELECT *
        FROM {table}
    """
    df = pd.read_sql_query(query, db_connection)
    # if profiling:
    #     profile = ProfileReport(df, title=f"{table} Profiling", explorative=True)
    #     profile.to_file(f"data/{table}_report.html")
    return df

def customer_view():
    query = """
        -- Vista consolidada de clientes
        CREATE VIEW customer_360 AS
        SELECT 
            c.customer_unique_id,
            c.customer_state,
            c.customer_city,
            COUNT(DISTINCT o.order_id) as total_orders,
            SUM(oi.price) as total_spent,
            AVG(r.review_score) as avg_review_score
        FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id
        LEFT JOIN order_items oi ON o.order_id = oi.order_id
        LEFT JOIN order_reviews r ON o.order_id = r.order_id
        GROUP BY 1,2,3;
    """
    df = pd.read_sql_query(query, db_connection)
    profile = ProfileReport(df, title="Customer View Profiling", explorative=True)
    profile.to_file("data/customer_view_report.html")
    return df

def print_unique_values(df, max_unique=50, exclude_patterns=None):
    """
    Prints unique values for all categorical columns in a DataFrame.
    Business question: What categories exist in the data and how are they distributed?
    Analysis: Explores categorical dimensions (e.g., payment methods, states, product categories)
    to understand the variety of values and prepare more detailed analyses.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to analyze
    max_unique : int, default=50
        Maximum number of unique values to display per column (to avoid printing too many)
    exclude_patterns : list, default=None
        List of string patterns to exclude from column names (e.g., ['_id', '_date', '_timestamp'])
    
    Returns:
    --------
    dict : Dictionary with column names as keys and unique values as values
    """
    # Default exclusion patterns
    if exclude_patterns is None:
        exclude_patterns = ['_id', '_date', '_timestamp', '_at', '_time']
    
    # Select only categorical columns (object or category types)
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Filter out columns based on exclusion patterns
    filtered_columns = []
    for col in categorical_columns:
        # Check if column name contains any exclusion pattern
        exclude = False
        for pattern in exclude_patterns:
            if pattern.lower() in col.lower():
                exclude = True
                break
        
        if not exclude:
            # Additional check: exclude if column appears to be datetime stored as string
            try:
                # Try to parse a sample of values as dates
                sample = df[col].dropna().head(10)
                if len(sample) > 0:
                    # Try converting to datetime - if it works for most values, it's likely a date column
                    pd.to_datetime(sample, errors='coerce')
                    if pd.to_datetime(sample, errors='coerce').notna().sum() / len(sample) > 0.8:
                        exclude = True
            except:
                pass
        
        if not exclude:
            filtered_columns.append(col)
    
    # Dictionary to store results
    unique_values_dict = {}
    
    # Print header
    print("=" * 60)
    print("CATEGORICAL COLUMNS ANALYSIS")
    print("=" * 60)
    
    if not filtered_columns:
        print("\nNo categorical columns found after filtering.")
        return unique_values_dict
    
    # Process each categorical column
    for col in filtered_columns:
        unique_values = df[col].dropna().unique()
        unique_count = len(unique_values)
        null_count = df[col].isna().sum()
        total_count = len(df)
        
        print(f"\n{'─' * 50}")
        print(f"Column: {col}")
        print(f"├─ Unique values: {unique_count}")
        print(f"├─ Null values: {null_count} ({null_count/total_count*100:.1f}%)")
        print(f"└─ Fill rate: {(total_count-null_count)/total_count*100:.1f}%")
        
        # Store in dictionary
        unique_values_dict[col] = unique_values
        
        # Show value counts for columns with reasonable number of unique values
        if unique_count <= 20:
            print(f"\nValue distribution:")
            value_counts = df[col].value_counts().head(10)
            for val, count in value_counts.items():
                percentage = count / total_count * 100
                bar = '█' * int(percentage / 2)  # Create simple bar chart
                print(f"  {val:20s}: {count:6d} ({percentage:5.1f}%) {bar}")
    
    print(f"\n{'=' * 60}")
    print(f"Total categorical columns analyzed: {len(filtered_columns)}")
    print(f"Total categorical columns excluded: {len(categorical_columns) - len(filtered_columns)}")
    print("=" * 60)
    
    return unique_values_dict


def analyze_column_types(df):
    """
    Provides a comprehensive analysis of all column types in a DataFrame.
    Business question: What is the structure and data types of our dataset?
    Analysis: Helps understand the data schema, identify potential data quality issues,
    and prepare appropriate processing strategies for each column type.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to analyze
    
    Returns:
    --------
    pandas.DataFrame : Summary DataFrame with column information
    """
    
    column_info = []
    
    for col in df.columns:
        info = {
            'column_name': col,
            'data_type': str(df[col].dtype),
            'null_count': df[col].isna().sum(),
            'null_percentage': f"{df[col].isna().sum() / len(df) * 100:.1f}%",
            'unique_count': df[col].nunique(),
            'unique_percentage': f"{df[col].nunique() / len(df) * 100:.1f}%"
        }
        
        # Add sample values for non-numeric columns
        if df[col].dtype == 'object':
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                info['sample_values'] = list(non_null_values.head(3))
            else:
                info['sample_values'] = []
        else:
            # For numeric columns, add statistics
            if pd.api.types.is_numeric_dtype(df[col]):
                info['min'] = df[col].min()
                info['max'] = df[col].max()
                info['mean'] = df[col].mean()
                info['median'] = df[col].median()
        
        # Identify potential column type
        col_lower = col.lower()
        if '_id' in col_lower:
            info['likely_type'] = 'ID'
        elif any(pattern in col_lower for pattern in ['_date', '_timestamp', '_at', '_time']):
            info['likely_type'] = 'DateTime'
        elif df[col].dtype == 'object' and df[col].nunique() / len(df) < 0.5:
            info['likely_type'] = 'Categorical'
        elif df[col].dtype == 'object' and df[col].nunique() / len(df) > 0.9:
            info['likely_type'] = 'Unique Text'
        elif pd.api.types.is_numeric_dtype(df[col]):
            info['likely_type'] = 'Numeric'
        else:
            info['likely_type'] = 'Other'
        
        column_info.append(info)
    
    summary_df = pd.DataFrame(column_info)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATAFRAME STRUCTURE ANALYSIS")
    print("=" * 60)
    print(f"\nTotal columns: {len(df.columns)}")
    print(f"Total rows: {len(df)}")
    print(f"\nColumn type distribution:")
    print(summary_df['likely_type'].value_counts())
    
    return summary_df


def get_categorical_columns(df, exclude_high_cardinality=True, cardinality_threshold=0.5):
    """
    Returns a list of categorical columns suitable for analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to analyze
    exclude_high_cardinality : bool, default=True
        Whether to exclude columns with too many unique values
    cardinality_threshold : float, default=0.5
        Threshold for unique values ratio (unique_values/total_rows)
    
    Returns:
    --------
    list : List of categorical column names
    """
    # Exclusion patterns
    exclude_patterns = ['_id', '_date', '_timestamp', '_at', '_time']
    
    # Get object and category columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Filter columns
    filtered_columns = []
    for col in categorical_columns:
        # Check exclusion patterns
        if any(pattern in col.lower() for pattern in exclude_patterns):
            continue
            
        # Check cardinality if requested
        if exclude_high_cardinality:
            cardinality = df[col].nunique() / len(df)
            if cardinality > cardinality_threshold:
                continue
        
        # Check if it's likely a datetime stored as string
        try:
            sample = df[col].dropna().head(10)
            if len(sample) > 0:
                parsed = pd.to_datetime(sample, errors='coerce')
                if parsed.notna().sum() / len(sample) > 0.8:
                    continue
        except:
            pass
        
        filtered_columns.append(col)
    
    return filtered_columns


# Example usage function
def explore_dataframe(df, name, detailed=False):
    """
    Comprehensive exploration of a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to explore
    detailed : bool, default=False
        Whether to show detailed analysis
    """
    print("\n" + "=" * 80)
    print(f"DATAFRAME EXPLORATION REPORT - {name.upper()}")
    print("=" * 80)
    
    # Basic info
    print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Column types analysis
    if detailed:
        print("\n" + "─" * 60)
        print("DETAILED COLUMN ANALYSIS")
        print("─" * 60)
        summary = analyze_column_types(df)
        print("\nColumn Details:")
        print(summary.to_string())
    
    # Categorical columns analysis
    print("\n" + "─" * 60)
    print("CATEGORICAL COLUMNS")
    print("─" * 60)
    categorical_cols = get_categorical_columns(df)
    print(f"Found {len(categorical_cols)} categorical columns suitable for analysis:")
    for col in categorical_cols:
        print(f"  • {col}")
    
    # Print unique values for categorical columns
    if categorical_cols:
        unique_values = print_unique_values(df[categorical_cols])
    
    return categorical_cols









