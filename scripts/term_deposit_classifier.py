"""
Training and validation script for term deposit classifier.

This module orchestrates the training pipeline for the term deposit classification
model. It performs data validation checks, tunes a Support Vector Classifier (SVC)
using randomized search, and persists the best performing model pipeline. Additionally,
it generates training performance artifacts including accuracy scores and confusion
matrices.

Author: Godsgift Braimah
Date: 2025-12-01
"""

import click
import os
import sys
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

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.feature_correlation import feature_corr
from src.random_search_svc import search_svc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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

    This function performs the following steps in the training pipeline:
    1. Loads the processed training data and preprocessor object.
    2. Executes custom feature correlation checks using Deepchecks.
    3. Performs hyperparameter tuning for an SVC model via RandomizedSearchCV.
    4. Serializes and saves the best model pipeline.
    5. Saves training accuracy scores and a confusion matrix plot to disk.

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
    
    Returns
    -------
    None
        This function does not return a value; it saves output files to disk.
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