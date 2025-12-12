"""
Feature correlation validation module.

This module provides functionality to validate the integrity of the training data
using the Deepchecks library. It specifically checks for excessive correlations
between features and the target label, as well as multicollinearity among features,
to ensure dataset quality before model training.

Author: Godsgift Braimah
Date: 2025-12-01
"""

import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureLabelCorrelation, FeatureFeatureCorrelation
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def feature_corr(df, target_col):
    """
    Runs Deepchecks validation for feature-label and feature-feature correlations.

    This function converts the input DataFrame into a Deepchecks Dataset, explicitly
    defining categorical features. It then validates two conditions:
    1. The Predictive Power Score (PPS) between any feature and the label must be < 0.9.
    2. The correlation between any pair of features must be < 0.92.

    Parameters
    ----------
    df : pd.DataFrame
        The training DataFrame containing features and the target.
    target_col : str
        The name of the target column.
    
    Returns
    -------
    None
        Returns None if all checks pass.
        
    Raises
    ------
    ValueError
        If the Feature-Label correlation or Feature-Feature correlation
        exceeds the maximum acceptable thresholds.
    """
    # Pass input as a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame, but got a " + str(type(df)))
    
    # Initialize Deepchecks Dataset
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