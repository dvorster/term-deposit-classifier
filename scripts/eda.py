"""Exploratory Data Analysis Script for Bank Marketing Dataset.

This script performs exploratory data analysis on the bank marketing dataset,
including data splitting, visualization generation, and saving processed data.

The script generates three types of visualizations:
- Numeric variable distributions
- Categorical variable distributions
- Correlation plot for numeric variables
"""
import click
import os
import pandas as pd
import altair_ally as aly
from sklearn.model_selection import train_test_split
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.create_visualizations import create_visualizations


# split data
@click.command()
@click.option(
    "--loaded-data", 
    type=str, 
    help="Path to loaded data"
)
@click.option(
    "--processed-data",
    type=str,
    required=True,
    help="Path to processed training data",
)
@click.option(
    "--plot-to",
    required=True,
    type=str,
    help="Path to directory where the plot will be written to",
)


def main(loaded_data, processed_data, plot_to):
    """Run exploratory data analysis and generate visualizations.

    This function loads the bank marketing dataset, splits it into training and
    test sets, performs basic data exploration, generates visualizations for
    numeric and categorical variables, and saves all outputs to specified paths.

    Parameters
    ----------
    loaded_data : str
        Path to the input CSV file containing the raw bank marketing data as a pandas DataFrame.
    processed_data : str
        Directory path where the processed train and test CSV files as pandas DataFrames will be saved.
    plot_to : str
        Directory path where the generated visualization plots will be saved.

    Returns
    -------
    None
        The function saves outputs to disk and does not return any values.

    Notes
    -----
    - Uses stratified train-test split with 80/20 ratio
    - Random state is set to 522 for reproducibility
    - Generates three plots: numeric distributions, categorical distributions,
      and correlation matrix
    - All output directories are created automatically if they don't exist

    Examples
    --------
    >>> main(
         loaded_data='data/raw/bank_marketing.csv',
         processed_data='data/processed',
         plot_to='results/figures'
     )
    """
    # Create output directories
    os.makedirs(processed_data, exist_ok=True)
    os.makedirs(plot_to, exist_ok=True)

    # Load Data
    bank_marketing_sample = pd.read_csv(
        loaded_data, index_col=0)

    train_df, test_df = train_test_split(
        bank_marketing_sample,
        test_size=0.2,
        stratify=bank_marketing_sample["y"],
        random_state=522,
    )
    (train_df.isnull().sum() / train_df.shape[0]) * 100
    train_df.info()

    # Saves the data
    train_path = os.path.join(processed_data, "train.csv")
    test_path = os.path.join(processed_data, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    create_visualizations(train_df, bank_marketing_sample, plot_to)

    print("All tasks completed!")


if __name__ == '__main__':
    main()
