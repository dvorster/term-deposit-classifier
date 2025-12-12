"""
Tests for Feature-Label and Feature-Feature Correlation.

This module provides functionality to test if the logic for
our function: `feature_corr` which checks correlations between 
features and the target label in our data.

Author: Godsgift Braimah
Date: 2025-12-12
"""
import pytest
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.feature_correlation import feature_corr

@pytest.fixture
def sample_validation_df():
    """
    Creates sample  DataFrame matching for Deepchecks validation.
    Includes all categorical columns expected by feature_corr.
    """
    np.random.seed(42)  # For reproducibility
    n_rows = 50

    df = pd.DataFrame({
        'age': np.random.randint(20, 80, n_rows),
        'job': np.random.choice(['blue-collar', 'entrepreneur', 'technician', 'management', 'retired'], n_rows),
        'marital': np.random.choice(['married', 'single', 'divorced'], n_rows),
        'education': np.random.choice(['primary', 'secondary', 'tertiary', 'unknown'], n_rows),
        'default': np.random.choice(['no', 'yes'], n_rows, p=[0.9, 0.1]),
        'balance': np.random.randint(-100, 10000, n_rows),
        'housing': np.random.choice(['yes', 'no'], n_rows),
        'loan': np.random.choice(['yes', 'no'], n_rows),
        'contact': np.random.choice(['cellular', 'telephone', 'unknown'], n_rows),
        'month': np.random.choice(['jan', 'feb', 'mar', 'apr', 'may', 'jun'], n_rows),
        'duration': np.random.randint(0, 1000, n_rows),
        'campaign': np.random.randint(1, 10, n_rows),
        'previous': np.random.randint(0, 5, n_rows),
        'pdays_contacted': np.random.choice(['never', 'failure', 'success'], n_rows),
        'target': np.random.randint(0, 2, n_rows)
    })
    # df = pd.DataFrame({
    #     'age': [27, 29, 51, 34] * 5,
    #     'job': ['blue-collar', 'entrepreneur', 'blue-collar', 'management'] * 5,
    #     'marital': ['married', 'married', 'married', 'divorced'] * 5,
    #     'education': ['primary', 'secondary', 'primary', 'tertiary'] * 5,
    #     'default': ['no', 'no', 'no', 'no'] * 5,
    #     'balance': [237, 926, 507, 4231] * 5,
    #     'housing': ['yes', 'yes', 'yes', 'yes'] * 5,
    #     'loan': ['no', 'no', 'no', 'no'] * 5,
    #     'contact': ['cellular', 'cellular', 'cellular', 'cellular'] * 5,
    #     'month': ['jul', 'jul', 'aug', 'aug'] * 5,
    #     'duration': [283, 169, 85, 382] * 5,
    #     'campaign': [6, 2, 2, 2] * 5,
    #     'previous': [0, 0, 0, 0] * 5,
    #     'pdays_contacted': ['never', 'never', 'never', 'never'] * 5,
    #     'target': [0, 0, 0, 0] * 5
    # })
    
    # Adds variation to target and one feature to ensure correlations can be calculated and tested
    #df.loc[0:5, 'target'] = 1 
    #df.loc[0:5, 'age'] = 60
    return df

def test_feature_corr_success(sample_validation_df):
    """
    Test that feature_corr runs successfully on data the matching schema.
    """
    try:
        # Should return None and print "Data validation checks passed."
        feature_corr(sample_validation_df, 'target')
    except ValueError as e:
        pytest.fail(f"feature_corr raised ValueError unexpectedly: {e}")
    except KeyError as e:
        pytest.fail(f"feature_corr failed due to missing columns: {e}")


def test_feature_label_correlation_error(sample_validation_df):
    """
    Test that feature_corr raises a ValueError if we have a high correlation
    between the target and a feature.
    """
    # Forcing the 'age' column to be identical to 'target' to induce correlation
    sample_validation_df['age'] = sample_validation_df['target']
    
    # Raise error if correlation exceeds 0.92
    with pytest.raises(ValueError, match="Feature-Label correlation exceeds"):
        feature_corr(sample_validation_df, 'target')


def test_feature_feature_correlation_error(sample_validation_df):
    """
    Test that feature_corr raises a ValueError if two features are highly correlated.
    """
    # Force 'balance' to be identical to 'age' (Feature-Feature correlation)
    sample_validation_df['balance'] = sample_validation_df['age']
    
    # Check specifically for the "Feature-feature" error message
    with pytest.raises(ValueError, match="Feature-feature correlation exceeds"):
        feature_corr(sample_validation_df, 'target')


def test_feature_corr_edge_case_empty():
    """Test that function raises an error (ValueError) when input is empty."""
    empty_df = pd.DataFrame(columns=['age', 'job', 'marital', 'education', 'default', 
                                     'housing', 'loan', 'contact', 'month', 'pdays_contacted', 'target'])
    # Raise an error for empty datasets
    with pytest.raises((ValueError, Exception)):
        feature_corr(empty_df, 'target')


def test_feature_corr_wrong_type():
    """Test that function raises an error (TypeError/AttributeError) when input is not a DataFrame."""
    invalid_input = "This is not a dataframe"
    
    # Raise an error for Wrong input
    with pytest.raises((TypeError, AttributeError, ValueError)):
        feature_corr(invalid_input, 'target')