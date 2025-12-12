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

def feature_corr(df, target_col):
    """
    Runs Deepchecks validation for feature-label and feature-feature correlations.

    Converts the DataFrame to a Deepchecks Dataset (specifying categorical features)
    and checks if the Predictive Power Score (PPS) between features and labels
    is less than 0.9, and if feature-feature correlations are below 0.92.

    Parameters
    ----------
    df : pd.DataFrame
        The training DataFrame containing features and the target.
    target_col : str
        The name of the target column.

    Raises
    ------
    ValueError
        If the Feature-Label correlation or Feature-Feature correlation
        exceeds the maximum acceptable thresholds.
    """
    ds = Dataset(df, label=target_col, cat_features=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'pdays_contacted'])

    # Check feature-label correlations
    check_feat_lab = FeatureLabelCorrelation().add_condition_feature_pps_less_than(0.9)
    result_feat_lab = check_feat_lab.run(dataset=ds)

    # Check feature-feature correlations
    check_feat_feat = FeatureFeatureCorrelation().add_condition_max_number_of_pairs_above_threshold(
        threshold=0.92, n_pairs=0
    )
    result_feat_feat = check_feat_feat.run(dataset=ds)

    if not result_feat_lab.passed_conditions():
        raise ValueError("Feature-Label correlation exceeds the maximum acceptable threshold.")

    if not result_feat_feat.passed_conditions():
        raise ValueError("Feature-feature correlation exceeds the maximum acceptable threshold.")
    
    print("Data validation checks passed.")

    return None