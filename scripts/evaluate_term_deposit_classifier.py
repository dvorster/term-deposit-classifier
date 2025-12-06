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

@click.command()
@click.option('--test-data', type=str, help="Path to test data CSV")
@click.option('--preprocessor', type=str, help="Path to preprocessor pickle object")
@click.option('--columns-to-drop', type=str, help="Optional: Path to columns to drop from data")
@click.option('--pipeline-from', type=str, help="Path to the Directory where the pipeline was saved")
@click.option('--plot-to', type=str, help="Directory to save the plots")
@click.option('--target-col', type=str, default='target', help="Name of the target/label column")
@click.option('--seed', type=int, default=522, help="Random seed")