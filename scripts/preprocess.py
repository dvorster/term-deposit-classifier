
import os
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

@click.command()
@click.option('--train-csv-file', type=str, help="Path to raw train data")
@click.option('--test-csv-file', type=str, help="Path to raw test data")
@click.option('--data-to', type=str, help="Path to directory where processed data will be written to")
@click.option('--preprocessor-to', type=str, help="Path to directory where the preprocessor object will be written to")
@click.option('--plot-to', type=str, help="Path to directory where the chart will be written to")
def main(train_csv_file, test_csv_file, data_to, preprocessor_to, plot_to):
    """
    Performs validation, preprocessing and exploratory analysis.

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
    # map the target variable to numeric
    train_df['y'] = train_df['y'].map({'yes': 1, 'no': 0})

    # feature engineering on 'pdays' column into categorical determining if client was contacted before or not
    train_df['pdays_contacted'] = train_df['pdays'].apply(lambda x: 'never' if x == -1 else 'contacted')

    # Drop columns that are not needed from EDA: poutcome has 83% missing values.
    drop_cols = ['day_of_week', 'pdays', 'poutcome']
    train_df = train_df.drop(columns=drop_cols)

    # split data
    X_train = train_df.drop(columns='y')
    y_train = train_df['y']

    # Rename target column for Deepchecks
    train_df.rename(columns={'y': 'target'}, inplace=True)

    # create Deepchecks Dataset
    X_train_ds = Dataset(train_df, label="target", cat_features=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
        'month', 'pdays_contacted'])

    # Outlier Detection
    check_outliers = OutlierSampleDetection()
    check_outliers.add_condition_outlier_ratio_less_or_equal(0.05)
    result_outliers = check_outliers.run(X_train_ds)

    # Single Value Check
    check_single_val = IsSingleValue()
    result_single_val = check_single_val.run(X_train_ds)

    # String Mismatch Check
    check_string_mismatch = StringMismatch()
    result_string_mismatch = check_string_mismatch.run(X_train_ds)

    # Class Imbalance Check
    check_imbalance = ClassImbalance()
    check_imbalance.add_condition_class_ratio_less_than(0.99)
    result_imbalance = check_imbalance.run(X_train_ds)

    result_checks = {
                'Outliers': result_outliers,
                'Single Value': result_single_val,
                'String Mismatch': result_string_mismatch,
                'Class Imbalance': result_imbalance
        }

    for name, result in result_checks.items():
        if not result.passed_conditions():
                raise ValueError(f"Check '{name}' failed!!")
        else:
                print(f"Check '{name}' passed.")

    ### The following code is for test data. ###
    test_df = pd.read_csv(test_csv_file)
    # map the target variable to numeric
    test_df['y'] = test_df['y'].map({'yes': 1, 'no': 0})

    # feature engineering on 'pdays' column into categorical determining if client was contacted before or not
    test_df['pdays_contacted'] = test_df['pdays'].apply(lambda x: 'never' if x == -1 else 'contacted')

    # Drop columns that are not needed from EDA: poutcome has 83% missing values.
    drop_cols = ['day_of_week', 'pdays', 'poutcome']
    test_df = test_df.drop(columns=drop_cols)

    # split data
    X_test = test_df.drop(columns='y')
    y_test = test_df['y']

    # Rename target column for Deepchecks
    test_df.rename(columns={'y': 'target'}, inplace=True)

    # create Deepchecks Dataset
    X_test_ds = Dataset(test_df, label="target", cat_features=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
        'month', 'pdays_contacted'])

    # Outlier Detection
    check_outliers = OutlierSampleDetection()
    check_outliers.add_condition_outlier_ratio_less_or_equal(0.05)
    result_outliers = check_outliers.run(X_test_ds)

    # Single Value Check
    check_single_val = IsSingleValue()
    result_single_val = check_single_val.run(X_test_ds)

    # String Mismatch Check
    check_string_mismatch = StringMismatch()
    result_string_mismatch = check_string_mismatch.run(X_test_ds)

    # Class Imbalance Check
    check_imbalance = ClassImbalance()
    check_imbalance.add_condition_class_ratio_less_than(0.99)
    result_imbalance = check_imbalance.run(X_test_ds)

    result_checks = {
                'Outliers': result_outliers,
                'Single Value': result_single_val,
                'String Mismatch': result_string_mismatch,
                'Class Imbalance': result_imbalance
        }

    for name, result in result_checks.items():
        if not result.passed_conditions():
                raise ValueError(f"Check '{name}' failed!!")
        else:
                print(f"Check '{name}' passed.")
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

    train_df.to_csv(os.path.join(data_to, "preprocess_train.csv"), index=False)
    test_df.to_csv(os.path.join(data_to, "preprocess_test.csv"), index=False)

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