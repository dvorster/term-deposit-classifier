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