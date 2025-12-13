"""
Preprocess script for term deposit classifier.

This script performs data validation, feature engineering,
and saves the processed data and preprocessor pipeline. It also generates
a correlation heatmap for numerical features.

Author: Teem KWONG
Date: 2025-12-13
"""

import os
import sys
import pickle
import altair as alt
import pandas as pd
import pandera as pa
import click
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import *
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocess_deepcheck import preprocess_deepcheck

@click.command()
@click.option('--train-csv-file', type=str, help="Path to raw train data")
@click.option('--test-csv-file', type=str, help="Path to raw test data")
@click.option('--data-to', type=str, help="Path to directory where processed data will be written to")
@click.option('--preprocessor-to', type=str, help="Path to directory where the preprocessor object will be written to")
@click.option('--plot-to', type=str, help="Path to directory where the chart will be written to")
def main(train_csv_file, test_csv_file, data_to, preprocessor_to, plot_to):
    """
    Performs validation, preprocessing and exploratory analysis.

    This function reads the raw training and testing data, performs initial data
    validation and feature engineering, fits a ColumnTransformer preprocessor
    on the training data, transforms both datasets,
    and saves the results. It also performs correlation checks
    and generates a numerical feature correlation heatmap.

    Parameters
    ----------
    train_csv_file : str
        Path to raw train data.
    test_csv_file : str
        Path to raw test data.
    data_to : str
        Path to directory where processed data will be written to.
    preprocessor_to : str
        Path to directory where the preprocessor object will be written to.
    plot_to : str
        Path to directory where the chart will be written to.

    Returns
    -------
    None
        The function saves processed files and the preprocessor pickle file.

    Raises
    ------
    ValueError
        If any of the Deepchecks data validation conditions fail.
    """
    ############################################################
    ### The following code is for BOTH train and test data. ###
    ############################################################
    ### The following code is for train data. ###
    # preprocessing
    train_df = pd.read_csv(train_csv_file)
    processed_train_df, X_train, y_train = preprocess_deepcheck(train_df)

    ### The following code is for test data. ###
    test_df = pd.read_csv(test_csv_file)
    processed_test_df, X_test, y_test = preprocess_deepcheck(test_df)
    ############################################################
    ### END ###
    ############################################################

    # separating columns by type of transformation required
    # One-hot encoding
    categorical_cols = ['job', 'marital', 'default', 'housing', 'loan', 'contact','month', 'pdays_contacted']
    # Ordinal encoding
    ordinal_cols = ['education']
    # Standard scaling
    numerical_cols = ['age', 'balance', 'duration', 'campaign', 'previous']

    # defining the preprocessor
    data_preprocessor = make_column_transformer(
    (
        make_pipeline(SimpleImputer(strategy='most_frequent'), 
                      OneHotEncoder(handle_unknown='ignore')), 
                      categorical_cols
    ), (
        make_pipeline(SimpleImputer(strategy='most_frequent'), 
                      OrdinalEncoder(categories=[['unknown', 'primary', 'secondary', 'tertiary']], dtype=object)), 
                      ordinal_cols
    ), (
        StandardScaler(), numerical_cols
        )
    )
    pickle.dump(data_preprocessor, open(os.path.join(preprocessor_to, "data_preprocessor.pickle"), "wb"))

    ############################################################
    ### The following code is for BOTH train and test data. ###
    ############################################################
    # Code adapted from from Tiffany A. Timbers, Joel Ostblom & Melissa Lee 2023/11/09: Breast Cancer Predictor Report
    data_preprocessor.fit(X_train)
    scaled_X_train = data_preprocessor.transform(X_train)
    scaled_X_test = data_preprocessor.transform(X_test)

    col_names = data_preprocessor.get_feature_names_out()

    scaled_X_train_df = pd.DataFrame(scaled_X_train, columns=col_names)
    scaled_X_test_df = pd.DataFrame(scaled_X_test, columns=col_names)

    scaled_X_train_df.to_csv(os.path.join(data_to, "scaled_train.csv"), index=False)
    scaled_X_test_df.to_csv(os.path.join(data_to, "scaled_test.csv"), index=False)

    processed_train_df.to_csv(os.path.join(data_to, "preprocess_train.csv"), index=False)
    processed_test_df.to_csv(os.path.join(data_to, "preprocess_test.csv"), index=False)

    ######################################################
    ### The following code is for train data ONLY. ###
    ######################################################
    col_names = data_preprocessor.named_transformers_['pipeline-1'].get_feature_names_out().tolist() + ordinal_cols + numerical_cols
    scaled_X_train_df = pd.DataFrame(scaled_X_train, columns=col_names)

    correlation_matrix = scaled_X_train_df[numerical_cols].corr()
    correlation_long = correlation_matrix.reset_index().melt(id_vars='index')
    correlation_long.columns = ['Feature 1', 'Feature 2', 'Correlation']

    corr_plot = alt.Chart(correlation_long).mark_rect().encode(
        x='Feature 1:O',
        y='Feature 2:O',
        color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['Feature 1', 'Feature 2', 'Correlation']
    ).properties(
        width=400,
        height=400,
        title="Correlation Heatmap"
    )

    corr_plot.save(os.path.join(plot_to, "correlation_heat_map.png"),
              scale_factor=2.0)

    # Code adapted from from Tiffany A. Timbers, Joel Ostblom & Melissa Lee 2023/11/09: Breast Cancer Predictor Report
    # Data validation checks: feature-target and feature-feature correlations
    # Check feature-label correlations

    X_train_ds = Dataset(processed_train_df, label="target", cat_features=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
        'month', 'pdays_contacted'])

    check_feat_lab_corr = FeatureLabelCorrelation().add_condition_feature_pps_less_than(0.9)
    check_feat_lab_corr_result = check_feat_lab_corr.run(dataset=X_train_ds)

    # Check feature-feature correlations
    check_feat_feat_corr = FeatureFeatureCorrelation().add_condition_max_number_of_pairs_above_threshold(threshold = 0.92, n_pairs = 0)
    check_feat_feat_corr_result = check_feat_feat_corr.run(dataset=X_train_ds)

    if not check_feat_lab_corr_result.passed_conditions():
        raise ValueError("Feature-Label correlation exceeds the maximum acceptable threshold.")

    if not check_feat_feat_corr_result.passed_conditions():
        raise ValueError("Feature-feature correlation exceeds the maximum acceptable threshold.")
    
    ######################################################
    ### END ###
    ######################################################
    
if __name__ == '__main__':
    main()
