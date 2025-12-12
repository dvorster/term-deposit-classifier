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
    data = {
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
        'pdays': [999, 10, 999, 999, 5, 999, 999, 15, 999, 3], # Used for feature engineering
        'previous': [0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
        'poutcome': ['nonexistent', 'success', 'nonexistent', 'failure', 'success', 'nonexistent', 'failure', 'success', 'nonexistent', 'failure'], # Dropped
        'emp.var.rate': [1.1, -0.1, 1.1, -1.8, 1.1, -0.1, 1.1, -1.8, 1.1, -0.1],
        'cons.price.idx': [93.994, 93.918, 93.994, 92.893, 93.994, 93.918, 93.994, 92.893, 93.994, 93.918],
        'cons.conf.idx': [-36.4, -42.7, -36.4, -46.2, -36.4, -42.7, -36.4, -46.2, -36.4, -42.7],
        'euribor3m': [4.857, 4.20, 4.857, 1.25, 4.857, 4.20, 4.857, 1.25, 4.857, 4.20],
        'nr.employed': [5191.0, 5099.1, 5191.0, 5099.1, 5191.0, 5099.1, 5191.0, 5099.1, 5191.0, 5099.1],
        'y': ['no', 'yes', 'no', 'no', 'yes', 'no', 'no', 'yes', 'no', 'yes'] # Target column
    }
    return pd.DataFrame(data)

# --- Test Cases for Data Transformation ---

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
    assert y_target.iloc[1] == 1  # Original 'yes'
    assert y_target.iloc[0] == 0  # Original 'no'

def test_transformation_feature_engineering(base_df):
    """Test if 'pdays_contacted' is created correctly and 'pdays' is dropped."""
    target_df, X_target, _ = preprocess_deepcheck(base_df.copy())
    
    # Check for the new feature
    assert 'pdays_contacted' in X_target.columns
    
    # Check mapping logic: 999 should be 'never', others 'contacted'
    assert (X_target.loc[base_df['pdays'] == 999, 'pdays_contacted'] == 'never').all()
    assert (X_target.loc[base_df['pdays'] != 999, 'pdays_contacted'] == 'contacted').all()

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
        preprocess_deepcheck(base_df.copy())
    except ValueError as e:
        pytest.fail(f"Deepchecks validation unexpectedly failed: {e}")

def test_deepchecks_fails_on_outlier_ratio(base_df):
    """Test for failure due to excessive outliers (OutlierSampleDetection check)."""
    df_with_outliers = base_df.copy()
    
    # Create outliers by setting many rows to extreme values
    # The condition is > 5% outliers. We have 10 rows, so > 0.5 outliers.
    # We will make 1 outlier just to confirm the base_df passes, then more to fail.
    # The current setup is a bit tricky since Deepchecks defines outliers dynamically.
    
    # A safer way: drastically skew a column
    for i in range(1, 7): # Create 60% outliers
        df_with_outliers.loc[i, 'age'] = 1000 
    
    with pytest.raises(ValueError, match="Check 'Outliers' failed!!"):
        preprocess_deepcheck(df_with_outliers)

def test_deepchecks_fails_on_single_value(base_df):
    """Test for failure due to a single-value column (IsSingleValue check)."""
    df_single_value = base_df.copy()
    
    # Introduce a column where all values are the same
    df_single_value['single_val_feature'] = 'constant'
    
    with pytest.raises(ValueError, match="Check 'Single Value' failed!!"):
        preprocess_deepcheck(df_single_value)

def test_deepchecks_fails_on_string_mismatch(base_df):
    """Test for failure due to inconsistent string formatting (StringMismatch check)."""
    df_mismatch = base_df.copy()
    
    # Introduce a different case for a categorical feature
    df_mismatch.loc[1, 'job'] = 'MANAGEMENT' # Mix of 'management' and 'MANAGEMENT'
    
    with pytest.raises(ValueError, match="Check 'String Mismatch' failed!!"):
        preprocess_deepcheck(df_mismatch)

def test_deepchecks_fails_on_class_imbalance(base_df):
    """Test for failure due to severe class imbalance (ClassImbalance check)."""
    df_imbalance = base_df.copy()
    
    # The condition is a class ratio (min_class / max_class) less than 0.99.
    # In base_df (5 'yes', 5 'no'), the ratio is 5/5 = 1.0. This passes.
    
    # Change all but one 'yes' to 'no' (9 'no', 1 'yes'). Ratio is 1/9 ≈ 0.11. This fails.
    df_imbalance['y'] = 'no'
    df_imbalance.loc[0, 'y'] = 'yes' # Ensure there is at least one 'yes'
    
    with pytest.raises(ValueError, match="Check 'Class Imbalance' failed!!"):
        preprocess_deepcheck(df_imbalance)