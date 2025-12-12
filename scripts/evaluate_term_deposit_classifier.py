import click
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.stats import loguniform
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureLabelCorrelation, FeatureFeatureCorrelation
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

@click.command()
@click.option('--processed-test-data', type=str, help="Path to scaled test data CSV")
@click.option('--pipeline-from', type=str, help="Path to the Directory where the pipeline was saved")
@click.option('--plot-to', type=str, help="Directory to save the plots")
@click.option('--table-to', type=str, help="Directory to save the score table")
@click.option('--target-col', type=str, default='target', help="Name of the target/label column")
def main(processed_test_data, pipeline_from, plot_to, table_to, target_col):
    '''
    Evaluates the term deposit classifier on the test data and saves the results.

    Loads a pre-trained SVC model pipeline and processed test data. Calculates the
    model's accuracy score and generates a confusion matrix. The results are
    saved as a CSV file and a PNG image, respectively, in the specified output
    directories.

    Parameters
    ----------
    processed_test_data : str
        Path to the CSV file containing the processed test data.
    pipeline_from : str
        Path to the pickle file containing the trained model pipeline.
    plot_to : str
        Path to the directory where the confusion matrix plot will be saved.
    table_to : str
        Path to the directory where the accuracy score table will be saved.
    target_col : str, optional
        The name of the target class column in the dataframe. Default is 'target'.

    Returns
    -------
    None
    '''
    # Read Data
    test_df = pd.read_csv(processed_test_data)
    
    # Load Pipeline    
    with open(pipeline_from, "rb") as f:
        pipe = pickle.load(f)   

    # Prepare X and y
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Score on Test Data
    test_score = round(pipe.score(X_test, y_test), 4)
    test_score_df = pd.DataFrame({'metric':['accuracy'], 'score': [test_score]})
    
    # Create path and store file.
    score_path = os.path.join(table_to, "svc_test_score.csv")
    test_score_df.to_csv(score_path, index=False)
    print(f"Test score saved to {score_path}")

    # Classification Report on Test Data
    report = classification_report(y_test, pipe.predict(X_test), output_dict=True) 
    classification_report_df = pd.DataFrame(report).T.round(2)
    
    report_path = os.path.join(table_to, "svc_classification_report.csv")
    classification_report_df.to_csv(report_path, index=False)
    print(f"Classification Report saved to {report_path}")
    
    # Generate and Save Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(
        pipe,
        X_test,
        y_test,
        values_format="d"
    )
    plt.title("Test Data: Confusion Matrix for SVC model")
    
    plot_path = os.path.join(plot_to, "test_svc_confusion_matrix.png")
    plt.savefig(plot_path)
    print(f"Confusion matrix saved to {plot_path}")

if __name__ == '__main__':
    main()
