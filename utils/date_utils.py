"""
Date Utilities - Robust date parsing and cleaning
Handles common data quality issues like header rows mixed in data
"""

import pandas as pd
from datetime import datetime
import warnings

def clean_date_column(df, date_col, drop_invalid=True):
    """
    Clean and parse a date column with robust error handling
    
    Args:
        df: DataFrame containing the date column
        date_col: Name of the date column to clean
        drop_invalid: If True, drop rows where date parsing fails
    
    Returns:
        DataFrame with cleaned date column
    """
    if df is None or df.empty:
        return df
    
    if date_col not in df.columns:
        warnings.warn(f"Column '{date_col}' not found in DataFrame")
        return df
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # 1. Remove rows where the date column contains its own name (header rows mixed in data)
    df = df[df[date_col] != date_col]
    
    # 2. Remove null/empty values
    df = df[df[date_col].notna()]
    df = df[df[date_col] != '']
    
    # 3. Convert to datetime with error handling
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # 4. Optionally drop rows where conversion failed
    if drop_invalid:
        original_count = len(df)
        df = df[df[date_col].notna()]
        dropped = original_count - len(df)
        if dropped > 0:
            warnings.warn(f"Dropped {dropped} rows with invalid dates in '{date_col}'")
    
    return df


def clean_multiple_date_columns(df, date_columns, drop_invalid=True):
    """
    Clean multiple date columns in a DataFrame
    
    Args:
        df: DataFrame
        date_columns: List of date column names
        drop_invalid: If True, drop rows where any date parsing fails
    
    Returns:
        DataFrame with all date columns cleaned
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    for date_col in date_columns:
        if date_col in df.columns:
            df = clean_date_column(df, date_col, drop_invalid)
    
    return df


def safe_date_filter(df, date_col, start_date=None, end_date=None):
    """
    Safely filter DataFrame by date range
    
    Args:
        df: DataFrame
        date_col: Date column to filter on
        start_date: Start date (datetime or string)
        end_date: End date (datetime or string)
    
    Returns:
        Filtered DataFrame
    """
    if df is None or df.empty:
        return df
    
    if date_col not in df.columns:
        warnings.warn(f"Column '{date_col}' not found")
        return df
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = clean_date_column(df, date_col)
    
    # Apply filters
    if start_date is not None:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        df = df[df[date_col] >= start_date]
    
    if end_date is not None:
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        df = df[df[date_col] <= end_date]
    
    return df


def get_date_column_stats(df, date_col):
    """
    Get statistics about a date column
    
    Args:
        df: DataFrame
        date_col: Date column name
    
    Returns:
        Dictionary with date column statistics
    """
    if df is None or df.empty or date_col not in df.columns:
        return None
    
    stats = {
        'total_rows': len(df),
        'null_count': df[date_col].isna().sum(),
        'valid_dates': 0,
        'invalid_dates': 0,
        'earliest_date': None,
        'latest_date': None,
        'date_range_days': None
    }
    
    # Try to parse dates
    temp_df = df.copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors='coerce')
    
    stats['valid_dates'] = temp_df[date_col].notna().sum()
    stats['invalid_dates'] = stats['total_rows'] - stats['valid_dates'] - stats['null_count']
    
    if stats['valid_dates'] > 0:
        stats['earliest_date'] = temp_df[date_col].min()
        stats['latest_date'] = temp_df[date_col].max()
        
        if stats['earliest_date'] and stats['latest_date']:
            stats['date_range_days'] = (stats['latest_date'] - stats['earliest_date']).days
    
    return stats


# Example usage functions for common patterns

def prepare_orders_dataframe(orders_df):
    """
    Prepare orders DataFrame with cleaned date columns
    Common pattern for order data
    """
    date_columns = ['order_date', 'created_at', 'updated_at', 'shipped_date', 'delivered_date']
    return clean_multiple_date_columns(orders_df, date_columns, drop_invalid=True)


def prepare_customer_dataframe(customers_df):
    """
    Prepare customers DataFrame with cleaned date columns
    Common pattern for customer data
    """
    date_columns = ['created_date', 'registration_date', 'last_login', 'updated_at']
    return clean_multiple_date_columns(customers_df, date_columns, drop_invalid=True)


def prepare_transactions_dataframe(transactions_df):
    """
    Prepare transactions DataFrame with cleaned date columns
    Common pattern for transaction data
    """
    date_columns = ['transaction_date', 'created_at', 'processed_at', 'settled_at']
    return clean_multiple_date_columns(transactions_df, date_columns, drop_invalid=True)
