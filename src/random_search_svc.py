import click
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.stats import loguniform
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureLabelCorrelation, FeatureFeatureCorrelation
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def search_svc(X_train, y_train, preprocessor, seed):
    """
    Fits and tunes an SVC model using RandomizedSearchCV.

    Builds a pipeline with the provided preprocessor and an SVC classifier.
    Performs a randomized search over 'C' and 'gamma' using a log-uniform
    distribution.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        The feature matrix for training.
    y_train : pd.Series or np.ndarray
        The target vector for training.
    preprocessor : sklearn
        The preprocessor object to apply before the model.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    sklearn.model_selection.RandomizedSearchCV
        The fitted RandomizedSearchCV object containing the best estimator.
    """
    svc_pipe = make_pipeline(preprocessor, SVC(random_state=seed))
    
    param_dist = {
        "svc__C": loguniform(1e-2, 1e3),
        "svc__gamma": loguniform(1e-2, 1e3)
    }
    
    random_svc = RandomizedSearchCV(
        svc_pipe, 
        param_distributions=param_dist,
        n_iter=100, 
        n_jobs=-1, 
        return_train_score=True, 
        random_state=seed
    )
    
    random_svc.fit(X_train, y_train)

    return random_svc