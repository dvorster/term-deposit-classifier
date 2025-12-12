"""
Tests for Random Search Classifier.

This module provides functionality to test if the logic for
our function: `search_svc` which intializes our search for 
the best parameters of the SVC modle and saves our model pipeline.

Author: Godsgift Braimah
Date: 2025-12-12
"""
import pytest
import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import RandomizedSearchCV

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.random_search_svc import search_svc

@pytest.fixture
def test_training_data():
    """
    Creates simple numerical data to test the SVC search logic.
    """
    X = pd.DataFrame({
        'feat_A': np.random.rand(20), 
        'feat_B': np.random.rand(20)
    })
    y = pd.Series(np.random.randint(0, 2, 20), name='target')
    return X, y

@pytest.fixture
def test_preprocessor():
    """Returns a simple preprocessor (StandardScaler) for the SVC test."""
    return make_column_transformer(
        (StandardScaler(), ['feat_A', 'feat_B'])
    )

def test_search_svc_returns_fitted_model(test_training_data, test_preprocessor):
    """
    Test that search_svc runs, fits the model, and returns a RandomizedSearchCV object.
    """
    X_train, y_train = test_training_data
    seed = 42
    
    # Run the function
    model = search_svc(X_train, y_train, test_preprocessor, seed)
    
    # Assertions
    assert isinstance(model, RandomizedSearchCV)
    assert hasattr(model, "best_estimator_"), "Model should be fitted (have best_estimator_)"
    assert hasattr(model, "best_score_"), "Model should have a recorded best score"


def test_search_svc_empty_data(test_preprocessor):
    """
    Test with empty X and y.
    Sklearn's fit() raises ValueError if given empty dataframe.
    """
    X_empty = pd.DataFrame(columns=['feat_A', 'feat_B'])
    y_empty = pd.Series([], name='target')
    
    with pytest.raises(ValueError, match="Found array with 0 sample"):
        search_svc(X_empty, y_empty, test_preprocessor, seed=42)

def test_search_svc_single_class(test_preprocessor):
    """
    Test when target only has one class (e.g., all zeros).
    SVM requires at least 2 classes to define a boundary.
    """
    X = pd.DataFrame({'feat_A': np.random.rand(10), 'feat_B': np.random.rand(10)})
    y = pd.Series(np.zeros(10), name='target') # All targets are 0
    
    # Raises ValueError: "The number of classes has to be greater than one"
    with pytest.raises(ValueError):
        search_svc(X, y, test_preprocessor, seed=42)

def test_search_svc_wrong_type(test_preprocessor):
    """
    Abnormal Case: Input is a list instead of a DataFrame.
    """
    X_invalid = [[1, 2], [3, 4]]
    y_invalid = [0, 1]
    
    # Raises either KeyError, ValueError, or AttributeError if input not a DataFrame.
    with pytest.raises(Exception): 
        search_svc(X_invalid, y_invalid, test_preprocessor, seed=42)