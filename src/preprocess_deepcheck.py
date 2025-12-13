"""
Initial preprocessing and data validation script.

This script contains the `preprocess_deepcheck` function, which performs
initial feature engineering, drops unnecessary columns,
handles target variable encoding, and executes
a set of critical data integrity and validation checks using the
Deepchecks library. It is designed to be called by the main
data processing pipeline.

Author: Teem KWONG
Date: 2025-12-13
"""

import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import *

def preprocess_deepcheck(target_df):
    """
    Preprocess a target dataset and validate it using Deepchecks.

    The function performs data preprocessing, and creates a 
    Deepchecks `Dataset` object for validation. If any validation fails, an error is raised.

    Parameters
    ----------
    target_df : pandas.DataFrame
        Input dataframe containing the target column and features.

    Returns
    -------
    target_df : pandas.DataFrame
        Processed dataframe containing the target column and features.

    X_target : pandas.DataFrame
        Processed feature dataframe.

    y_target : pandas.Series
        Target dataframe.

    Raises
    ------
    TypeError
        If the target_df is not a pandas DataFrame.
    ValueError
        If required columns are missing, or any of the Deepchecks 
        data validation conditions fail.
    """

    # Check if target_df is a pandas DataFrame
    if not isinstance(target_df, pd.DataFrame):
        raise TypeError(f"Input 'target_df' must be a pandas DataFrame, but received type: {type(target_df)}")
    
    # Check for required columns before preprocessing
    required_cols = ['y', 'pdays', 'day_of_week', 'poutcome']
    missing_cols = [col for col in required_cols if col not in target_df.columns]
    
    if missing_cols:
        raise ValueError(f"Target DataFrame is missing required columns for preprocessing: {missing_cols}")
    
    # map the target variable to numeric
    target_df['y'] = target_df['y'].map({'yes': 1, 'no': 0})

    # feature engineering on 'pdays' column into categorical determining if client was contacted before or not
    target_df['pdays_contacted'] = target_df['pdays'].apply(lambda x: 'never' if x == -1 else 'contacted')

    # Drop columns that are not needed from EDA: poutcome has 83% missing values.
    drop_cols = ['day_of_week', 'pdays', 'poutcome']
    target_df = target_df.drop(columns=drop_cols)

    # split data
    X_target = target_df.drop(columns='y')
    y_target = target_df['y']

    # Rename target column for Deepchecks
    target_df.rename(columns={'y': 'target'}, inplace=True)

    # create Deepchecks Dataset
    X_target_ds = Dataset(target_df, label="target", cat_features=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
        'month', 'pdays_contacted'])

    # Outlier Detection
    check_outliers = OutlierSampleDetection()
    check_outliers.add_condition_outlier_ratio_less_or_equal(0.05)
    result_outliers = check_outliers.run(X_target_ds)

    # Single Value Check
    check_single_val = IsSingleValue()
    result_single_val = check_single_val.run(X_target_ds)

    # String Mismatch Check
    check_string_mismatch = StringMismatch()
    result_string_mismatch = check_string_mismatch.run(X_target_ds)

    # Class Imbalance Check
    check_imbalance = ClassImbalance()
    check_imbalance.add_condition_class_ratio_less_than(0.99)
    result_imbalance = check_imbalance.run(X_target_ds)

    result_checks = {
                'Outliers': result_outliers,
                'Single Value': result_single_val,
                'String Mismatch': result_string_mismatch,
                'Class Imbalance': result_imbalance
        }

    for name, result in result_checks.items():
        if not result.passed_conditions():
                raise ValueError(f"Check '{name}' failed!!")
        else:
                print(f"Check '{name}' passed.")

    return target_df, X_target, y_target
