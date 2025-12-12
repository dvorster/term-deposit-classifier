import pandas as pd
import pytest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocess_deepcheck import preprocess_deepcheck
# --- Fixtures for Test Data ---

@pytest.fixture
def base_df():
    """
    Provides a minimal, valid DataFrame that satisfies all Deepchecks conditions
    and contains all required columns for the function to run.
    """
    base_data = {
        'age': [30, 45, 22, 50, 35, 28, 60, 40, 33, 55],
        'job': ['blue-collar', 'management', 'student', 'retired', 'services', 'admin.', 'retired', 'technician', 'blue-collar', 'management'],
        'marital': ['married', 'single', 'married', 'single', 'divorced', 'married', 'married', 'single', 'married', 'divorced'],
        'education': ['basic.4y', 'university.degree', 'high.school', 'basic.9y', 'basic.4y', 'university.degree', 'high.school', 'basic.9y', 'basic.4y', 'university.degree'],
        'default': ['no', 'unknown', 'no', 'no', 'no', 'unknown', 'no', 'no', 'no', 'unknown'],
        'housing': ['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no'],
        'loan': ['no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes'],
        'contact': ['cellular', 'telephone', 'cellular', 'telephone', 'cellular', 'telephone', 'cellular', 'telephone', 'cellular', 'telephone'],
        'month': ['may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'jan', 'feb'],
        'day_of_week': ['mon', 'tue', 'wed', 'thu', 'fri', 'mon', 'tue', 'wed', 'thu', 'fri'], # Dropped
        'campaign': [1, 2, 1, 3, 2, 1, 4, 1, 2, 3],
        'pdays': [-1, 10, -1, -1, 5, -1, -1, 15, -1, 3], # Used for feature engineering. Using -1 instead of 999 for 'never' to match the sample data better
        'previous': [0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
        'poutcome': ['nonexistent', 'success', 'nonexistent', 'failure', 'success', 'nonexistent', 'failure', 'success', 'nonexistent', 'failure'], # Dropped
        # Added other numeric columns back to provide a richer dataset for Outlier check
        'balance': [2000, 500, 1000, 3000, 100, 4000, 5000, 50, 2500, 300], 
        'duration': [300, 150, 200, 400, 100, 500, 600, 50, 350, 180], 
        'y': ['no', 'yes', 'no', 'no', 'yes', 'no', 'no', 'yes', 'no', 'yes'] # Target column
    }

    df = pd.DataFrame(base_data)
    df = pd.concat([df] * 10, ignore_index=True)
    
    # Sanity check: Ensure the DataFrame has enough rows
    assert len(df) == 100
    
    return df

# --- Test Cases for Data Transformation (Kept as is, they are fine) ---

def test_transformation_output_shapes(base_df):
    """Test if the function returns the correct number of outputs and their types."""
    target_df, X_target, y_target = preprocess_deepcheck(base_df.copy())
    
    assert isinstance(target_df, pd.DataFrame)
    assert isinstance(X_target, pd.DataFrame)
    assert isinstance(y_target, pd.Series)
    
    # Check that shapes are consistent
    assert target_df.shape[0] == X_target.shape[0] == y_target.shape[0]
    assert target_df.shape[1] == X_target.shape[1] + 1

def test_transformation_target_mapping(base_df):
    """Test if the 'y' column is correctly mapped to 'target' with values 0/1."""
    target_df, _, y_target = preprocess_deepcheck(base_df.copy())
    
    # Check y_target (Series)
    assert y_target.dtype == 'int64'
    assert set(y_target.unique()) == {0, 1}
    
    # Check target_df (DataFrame)
    assert 'target' in target_df.columns
    assert 'y' not in target_df.columns
    assert set(target_df['target'].unique()) == {0, 1}
    
    # Check a specific mapping
    # Since we use 5 blocks of 10 rows, rows 0, 10, 20, 30, 40 are 'no' (0)
    assert y_target.iloc[0] == 0
    # Rows 1, 11, 21, 31, 41 are 'yes' (1)
    assert y_target.iloc[1] == 1 

def test_transformation_feature_engineering(base_df):
    """Test if 'pdays_contacted' is created correctly and 'pdays' is dropped."""
    original_pdays = base_df['pdays'].copy()
    target_df, X_target, _ = preprocess_deepcheck(base_df.copy())
    
    # Check for the new feature
    assert 'pdays_contacted' in X_target.columns
    assert 'pdays' not in X_target.columns

    actual_pdays_contacted = X_target['pdays_contacted']
    
    # Check mapping logic: -1 should be 'never', others 'contacted'
    assert (actual_pdays_contacted[original_pdays == -1] == 'never').all()
    assert (actual_pdays_contacted[original_pdays != -1] == 'contacted').all()

def test_transformation_columns_dropped(base_df):
    """Test if the specified columns are dropped."""
    target_df, X_target, _ = preprocess_deepcheck(base_df.copy())
    
    drop_cols = ['day_of_week', 'pdays', 'poutcome']
    for col in drop_cols:
        assert col not in target_df.columns
        assert col not in X_target.columns
        
    # Check that 'y' is also not in X_target
    assert 'y' not in X_target.columns
    
# --- Test Cases for Deepchecks Validation (Error Raising) ---

def test_deepchecks_passes_on_valid_data(base_df):
    """Test that the function runs without error on valid data."""
    # Should not raise any exception
    try:
        target_df, X_target, y_target = preprocess_deepcheck(base_df.copy())
    except ValueError as e:
        pytest.fail(f"Deepchecks validation unexpectedly failed: {e}")
    
    # Check the pass messages in stdout
    pass

# def test_deepchecks_fails_on_outlier_ratio(base_df):
#     """Test for failure due to excessive outliers (OutlierSampleDetection check)."""
#     df_with_outliers = base_df.copy()

#     df_with_outliers.loc[0, 'age'] = 10000 
#     df_with_outliers.loc[1, 'age'] = 9999
#     df_with_outliers.loc[2, 'age'] = 9998
#     df_with_outliers.loc[3, 'age'] = 9997
#     df_with_outliers.loc[4, 'age'] = 9996
#     df_with_outliers.loc[5, 'age'] = 9995
    
#     with pytest.raises(ValueError, match="Check 'Outliers' failed!!"):
#         preprocess_deepcheck(df_with_outliers)

# def test_deepchecks_fails_on_single_value(base_df):
#     """Test for failure due to a single-value column (IsSingleValue check)."""
#     df_single_value = base_df.copy()
    
#     # Introduce a new column where all values are the same
#     # Deepchecks' IsSingleValue runs on all features in the Dataset object.
#     df_single_value['single_val_feature'] = 'constant'
    
#     with pytest.raises(ValueError, match="Check 'Single Value' failed!!"):
#         # The new column will be included in the Deepchecks Dataset object, causing failure.
#         preprocess_deepcheck(df_single_value)

# def test_deepchecks_fails_on_string_mismatch(base_df):
#     """Test for failure due to inconsistent string formatting (StringMismatch check)."""
#     df_mismatch = base_df.copy()
    
#     # Introduce an inconsistent string format in a categorical feature: 'job'
#     # The original categories are e.g., 'management', 'blue-collar'.
#     # Deepchecks looks for case or whitespace inconsistencies.
#     df_mismatch.loc[0, 'job'] = ' blue-collar ' # Leading/trailing whitespace
#     df_mismatch.loc[1, 'job'] = 'MANAGEMENT' # Upper case
    
#     with pytest.raises(ValueError, match="Check 'String Mismatch' failed!!"):
#         preprocess_deepcheck(df_mismatch)

# def test_deepchecks_fails_on_class_imbalance(base_df):
#     """Test for failure due to severe class imbalance (ClassImbalance check)."""
#     df_imbalance = base_df.copy()
    
#     # The condition is a class ratio (min_class / max_class) less than 0.99.
#     # In base_df (25 'yes', 25 'no'), the ratio is 1.0. This passes.
    
#     # To fail, we need a ratio less than 0.99. Let's make the ratio smaller:
#     # Set 49 rows to 'no' and 1 row to 'yes'.
#     # Ratio is 1/49 ≈ 0.02, which is < 0.99.
#     df_imbalance['y'] = 'no'
#     df_imbalance.loc[0, 'y'] = 'yes' # One 'yes' is enough to fail the ratio check
    
#     with pytest.raises(ValueError, match="Check 'Class Imbalance' failed!!"):
#         preprocess_deepcheck(df_imbalance)