import sys
import os
# Add project root to sys.path (two levels up from this file)
current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
dataloader_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if dataloader_root not in sys.path:
    sys.path.insert(0, dataloader_root)


from datetime import datetime
import pandas as pd

####
## Time Preprocessing Functions
####
# All date data is by default converted to Unix timestamps (seconds since 1970-01-01)
# This is done to avoid issues with processing time as datetine obejects, as can be easily reversed

def _date_to_unix_timestamp(date_str: str, date_format="%d/%m/%Y") -> float:
    """
    Converts a date string to a Unix timestamp (seconds since 1970-01-01).
    """
    try:
        dt = datetime.strptime(date_str, date_format)
        return dt.timestamp()
    except Exception:
        return None  # or float('nan')

def convert_column_to_unix(df: pd.DataFrame, column_name: str, date_format="%d/%m/%Y") -> pd.Series:
    """
    Converts a date column in the DataFrame to Unix timestamps.
    Returns the converted Series.
    """
    return df[column_name].apply(lambda x: _date_to_unix_timestamp(str(x), date_format=date_format))

def _unix_timestamp_to_date(unix_timestamp: float, date_format="%d/%m/%Y") -> str:
    """
    Converts a Unix timestamp to a date string.
    """
    try:
        dt = datetime.fromtimestamp(unix_timestamp)
        return dt.strftime(date_format)
    except Exception:
        return None  # or float('nan')

def convert_unix_to_date(df: pd.DataFrame, column_name: str, date_format="%d/%m/%Y") -> pd.Series:
    """
    Converts a Unix timestamp column in the DataFrame to date strings.
    Returns the converted Series.
    """
    return df[column_name].apply(lambda x: _unix_timestamp_to_date(x, date_format=date_format))

####
## End of Time Preprocessing Functions
####

def _drop_empty_columns(data_df, threshold=0.5):
    """
    Drop columns with more than threshold missing values.
    """
    missing_values = data_df.isnull().mean()
    columns_to_drop = missing_values[missing_values > threshold].index
    
    # name the columns to be dropped
    print(f"Dropped: {columns_to_drop}")
    data_df = data_df.drop(columns=columns_to_drop)

    return data_df

def _drop_single_value_columns(data_df):
    """
    Drop columns with only one unique value.
    """
    unique_values = data_df.nunique()
    columns_to_drop = unique_values[unique_values == 1].index
    data_df = data_df.drop(columns=columns_to_drop)
    return data_df


####
## Data Preprocessing Functions
####
import re

def clean_numerical_string(value):
    if pd.isna(value):
        return value

    # Ensure it's string for regex processing
    value = str(value)

    # Keep only digits, dot, minus sign
    value_cleaned = re.sub(r'[^0-9\.\-]', '', value)

    # Handle multiple minus signs: keep only leading minus
    if '-' in value_cleaned:
        value_cleaned = '-' + value_cleaned.replace('-', '')

    # Handle multiple dots: keep only the rightmost one as decimal point
    if value_cleaned.count('.') > 1:
        parts = value_cleaned.split('.')
        value_cleaned = ''.join(parts[:-1]) + '.' + parts[-1]

    # Convert to float, fallback to NaN if fails
    try:
        return float(value_cleaned)
    except ValueError:
        return pd.NA
    
def clean_numerical_columns(summary_df: pd.DataFrame, dataset_files: list) -> pd.Series:

    # Get list of numerical columns based on summary_df 'Type'
    numerical_cols_from_summary = summary_df.loc[summary_df['Type'] == 'numerical', 'Column Name'].tolist()

    # Apply to numerical columns in df_file
    for df_file in dataset_files:
        for col in numerical_cols_from_summary:
            if col in df_file.columns:
                df_file[col] = df_file[col].apply(clean_numerical_string)

    # Update the summary_df with cleaned values
    for col in numerical_cols_from_summary:
        if col in df_file.columns:
            # Get the cleaned values from the dataframe
            cleaned_values = df_file[col].dropna().unique()
            # Update the summary_df with the cleaned values
            summary_df.loc[summary_df['Column Name'] == col, '# Categories'] = len(cleaned_values)
            if len(cleaned_values) > 0:
                summary_df.loc[summary_df['Column Name'] == col, 'Example Value'] = str(cleaned_values[0])
            else:
                summary_df.loc[summary_df['Column Name'] == col, 'Example Value'] = None
    
    return summary_df, dataset_files
