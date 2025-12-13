"""Tests for create_visualizations module.

This module contains tests for the create_visualizations function,
covering expected use cases, edge cases, and error handling scenarios.

Test Categories
---------------
- Simple/Expected Use Cases: Normal DataFrame inputs with mixed data types
- Edge Cases: Empty DataFrames, only numeric data, DataFrames with NaN values
- Abnormal/Error Cases: Invalid output paths, missing required columns

The test data files are stored in the same directory as this test file.
"""
import pytest
import pandas as pd
import os
import sys
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.create_visualizations import create_visualizations

# Define the path to test data directory
test_data_dir = os.path.dirname(__file__)

@pytest.fixture
def temp_output_dir(tmpdir):
    """Create a temporary directory for test outputs."""
    return str(tmpdir)

@pytest.fixture
def sample_dataframe():
    """Load sample DataFrame from CSV file for testing."""
    return pd.read_csv(os.path.join(test_data_dir, 'sample_data.csv'))

# Test for simple/expected use case
def test_visualizations_creation_simple(sample_dataframe, temp_output_dir):
    """
    Tests that visualization files are created successfully under normal conditions.
    """
    create_visualizations(sample_dataframe, sample_dataframe, temp_output_dir)
    
    assert os.path.exists(os.path.join(temp_output_dir, "numeric_univariate.png"))
    assert os.path.exists(os.path.join(temp_output_dir, "categorical_univariate.png"))
    assert os.path.exists(os.path.join(temp_output_dir, "correlation_plot.png"))

# Test for edge cases
def test_visualizations_with_empty_dataframe(temp_output_dir):
    """
    Tests the function's behavior with an empty DataFrame.
    It should handle the error gracefully without crashing.
    """
    empty_df = pd.read_csv(os.path.join(test_data_dir, 'empty_data.csv'))
    with pytest.raises(Exception) as excinfo:
        create_visualizations(empty_df, empty_df, temp_output_dir)
    assert "empty" in str(excinfo.value).lower()

def test_visualizations_with_only_numeric_data(temp_output_dir):
    """
    Tests that visualizations are created when the DataFrame contains only numeric data.
    """
    numeric_df = pd.read_csv(os.path.join(test_data_dir, 'numeric_only_data.csv'))
    create_visualizations(numeric_df, numeric_df, temp_output_dir)
    assert os.path.exists(os.path.join(temp_output_dir, "numeric_univariate.png"))
    assert os.path.exists(os.path.join(temp_output_dir, "correlation_plot.png"))
    # The categorical plot might not be created or be empty, which is acceptable.
    # We are primarily concerned that the function does not fail.

def test_visualizations_with_nan_values(temp_output_dir):
    """
    Tests the function's behavior with a DataFrame containing NaN values.
    """
    nan_df = pd.read_csv(os.path.join(test_data_dir, 'data_with_nan.csv'))
    create_visualizations(nan_df, nan_df, temp_output_dir)
    assert os.path.exists(os.path.join(temp_output_dir, "numeric_univariate.png"))
    assert os.path.exists(os.path.join(temp_output_dir, "categorical_univariate.png"))
    assert os.path.exists(os.path.join(temp_output_dir, "correlation_plot.png"))

# Test for abnormal/error cases
def test_visualizations_invalid_output_path(sample_dataframe):
    """
    Tests the function's error handling with an invalid output path.
    """
    # Using a path that is likely not writable
    invalid_path = "/non_existent_dir/plots"
    with pytest.raises(OSError):
        create_visualizations(sample_dataframe, sample_dataframe, invalid_path)

def test_visualizations_missing_target_column(temp_output_dir):
    """
    Tests error handling when the target column 'y' is missing.
    """
    df_no_y = pd.read_csv(os.path.join(test_data_dir, 'missing_y_column.csv'))
    with pytest.raises(KeyError):
        create_visualizations(df_no_y, df_no_y, temp_output_dir)
