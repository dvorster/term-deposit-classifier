#import packages

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

@click.command()
@click.option('--processed-train-data', type=str, help="Path to processed training data CSV")
@click.option('--preprocessor', type=str, help="Path to preprocessor pickle object")
@click.option('--pipeline-to', type=str, help="Directory to save the pipeline")
@click.option('--plot-to', type=str, help="Directory to save the plots")
@click.option('--table-to', type=str, help="Directory to save the score table")
@click.option('--target-col', type=str, default='target', help="Name of the target/label column")
@click.option('--seed', type=int, default=522, help="Random seed")
def main(processed_train_data, preprocessor, pipeline_to, plot_to, table_to, target_col, seed):
    '''
    Validates data, fits an SVC classifier, saves the pipeline, and saves artifacts.

    This function orchestrates the training pipeline:
    1. Loads processed training data and the preprocessor.
    2. Runs data validation checks (Deepchecks).
    3. Tunes an SVC model using RandomizedSearchCV.
    4. Saves the best model pipeline, training accuracy scores, and a confusion matrix plot.

    Parameters
    ----------
    processed_train_data : str
        Path to the CSV file containing the processed training data.
    preprocessor : str
        Path to the pickle file containing the preprocessor object.
    pipeline_to : str
        Directory path where the trained pipeline pickle file will be saved.
    plot_to : str
        Directory path where the confusion matrix plot will be saved.
    table_to : str
        Directory path where the training score CSV will be saved.
    target_col : str, optional
        The name of the target class column. Default is 'target'.
    seed : int, optional
        Random seed for reproducibility. Default is 522.
    '''
    # Read Data
    train_df = pd.read_csv(processed_train_data)

    with open(preprocessor, "rb") as f:
        data_preprocessor = pickle.load(f)
        
    # 1. Run Data Validation
    feature_corr(train_df, target_col)

    # Prepare X and y
    X_train = train_df.drop(columns=target_col, axis=1)
    y_train = train_df[target_col]


    # 2. Fit and Get the Best Parameters of the  Model
    print("Tuning SVC model")
    best_model = search_svc(X_train, y_train, data_preprocessor, seed)
    
    train_score = round(best_model.best_score_,4)
    train_score_df = pd.DataFrame({'metric':['accuracy'], 'score': [train_score]})
    
    # Check if table_to directory exists, if not create it
    os.makedirs(table_to, exist_ok=True)
    score_path = os.path.join(table_to, "svc_train_score.csv")
    train_score_df.to_csv(score_path, index=False)
    print(f"Train score saved to {score_path}")


    # 3. Save the Model
    os.makedirs(pipeline_to, exist_ok=True)
    model_path = os.path.join(pipeline_to, "svc_pipeline.pickle")
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Model saved to {model_path}")

    # 4. Generate and Save Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(
        best_model,
        X_train,
        y_train,
        values_format="d"
    )
    plt.title("Train Data: Confusion Matrix for SVC model")
    
    plot_path = os.path.join(plot_to, "train_svc_confusion_matrix.png")
    plt.savefig(plot_path)
    print(f"Confusion matrix saved to {plot_path}")

if __name__ == '__main__':
    main()