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

<<<<<<< HEAD
<<<<<<< HEAD
=======
# Filter warnings to keep the output clean
>>>>>>> f693be3 (Added term deposit classification script)
=======
>>>>>>> fee509b (Fixed imported packages)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def feature_corr(df, target_col):
    """
    Runs Deepchecks validation for feature-label and feature-feature correlations.
    Raises ValueError if thresholds are exceeded.
    """
<<<<<<< HEAD
    ds = Dataset(df, label=target_col, cat_features=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
       'month', 'pdays_contacted'])
=======
    ds = Dataset(df, label=target_col, cat_features=[])
>>>>>>> f693be3 (Added term deposit classification script)

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
    Returns the fitted search object.
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
<<<<<<< HEAD
#@click.option('--train-data', type=str, help="Path to training data CSV")
@click.option('--processed-train-data', type=str, help="Path to processed training data CSV")
@click.option('--preprocessor', type=str, help="Path to preprocessor pickle object")
@click.option('--pipeline-to', type=str, help="Directory to save the pipeline")
@click.option('--plot-to', type=str, help="Directory to save the plots")
@click.option('--table-to', type=str, help="Directory to save the score table")
@click.option('--target-col', type=str, default='target', help="Name of the target/label column")
@click.option('--seed', type=int, default=522, help="Random seed")
def main(processed_train_data, preprocessor, pipeline_to, plot_to, table_to, target_col, seed):
=======
@click.option('--train-data', type=str, help="Path to training data CSV")
@click.option('--preprocessor', type=str, help="Path to preprocessor pickle object")
@click.option('--columns-to-drop', type=str, help="Optional: Path to columns to drop from data")
@click.option('--pipeline-to', type=str, help="Directory to save the pipeline")
@click.option('--plot-to', type=str, help="Directory to save the plots")
@click.option('--table-to', type=str, help="Directory to save the score table")
@click.option('--target-col', type=str, default='target', help="Name of the target/label column")
@click.option('--seed', type=int, default=522, help="Random seed")
<<<<<<< HEAD
def main(train_data, preprocessor, columns_to_drop, pipeline_to, plot_to, target_col, seed):
>>>>>>> f693be3 (Added term deposit classification script)
=======
def main(train_data, preprocessor, columns_to_drop, pipeline_to, plot_to, table_to, target_col, seed):
>>>>>>> 0e0d030 (Updated Classifiier Scripts)
    '''
    Validates data, fits an SVC classifier, saves the pipeline, 
    and saves a confusion matrix plot.
    '''
<<<<<<< HEAD
<<<<<<< HEAD
    # Read Data
    train_df = pd.read_csv(processed_train_data)
=======
    # Load resources
=======
    # Read Data
>>>>>>> 0e0d030 (Updated Classifiier Scripts)
    train_df = pd.read_csv(train_data)
>>>>>>> f693be3 (Added term deposit classification script)

    with open(preprocessor, "rb") as f:
        data_preprocessor = pickle.load(f)
        
<<<<<<< HEAD
    # 1. Run Data Validation
    # We pass the whole dataframe because Deepchecks needs the label column context
    #feature_corr(train_df, target_col)
=======
    if columns_to_drop:
        to_drop = pd.read_csv(columns_to_drop).features.tolist()
        train_df = train_df.drop(columns=to_drop)

    # 1. Run Data Validation
    # We pass the whole dataframe because Deepchecks needs the label column context
    feature_corr(train_df, target_col)
>>>>>>> f693be3 (Added term deposit classification script)

    # Prepare X and y
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]


    # 2. Fit and Get the Best Parameters of the  Model
    print("Tuning SVC model")
    best_model = search_svc(X_train, y_train, data_preprocessor, seed)
    
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 0e0d030 (Updated Classifiier Scripts)
    train_score = round(best_model.best_score_,4)
    train_score_df = pd.DataFrame({'metric':['accuracy'], 'score': [train_score]})
    
    # Check if table_to directory exists, if not create it
    os.makedirs(table_to, exist_ok=True)
    score_path = os.path.join(table_to, "svc_train_score.csv")
    train_score_df.to_csv(score_path, index=False)
<<<<<<< HEAD
=======
    print(f'Best Train Score: {best_model.best_score_:.3f}')
>>>>>>> f693be3 (Added term deposit classification script)
=======
>>>>>>> 0e0d030 (Updated Classifiier Scripts)


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
<<<<<<< HEAD
<<<<<<< HEAD
    plt.title("Train Data: Confusion Matrix for SVC model")
    
    plot_path = os.path.join(plot_to, "train_svc_confusion_matrix.png")
=======
    plt.title("Figure 5: Confusion Matrix for SVC model")
    
    plot_path = os.path.join(plot_to, "svc_confusion_matrix.png")
>>>>>>> f693be3 (Added term deposit classification script)
=======
    plt.title("Train Data: Confusion Matrix for SVC model")
    
    plot_path = os.path.join(plot_to, "train_svc_confusion_matrix.png")
>>>>>>> fee509b (Fixed imported packages)
    plt.savefig(plot_path)
    print(f"Confusion matrix saved to {plot_path}")

if __name__ == '__main__':
    main()