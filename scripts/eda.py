import click
import os
import pandas as pd
import altair_ally as aly
from sklearn.model_selection import train_test_split

# from load_data import load_processed_data


# split data
@click.command()
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
def main(processed_data, plot_to):
    """Run EDA, create plots, split data and save outputs"""
    # Create output directories
    os.makedirs(processed_data, exist_ok=True)
    os.makedirs(plot_to, exist_ok=True)

    # Load Data
    bank_marketing_sample = pd.read_csv(
        "data/raw/bank_marketing_sample.csv", index_col=0
    )

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

    numeric_plot = aly.dist(train_df, color="y")

    numeric_plot.properties(
        title="Figure 1: Univariate distributions of numeric variables in Bank Marketing Dataset"  # noqa: E501
    )
    numeric_plot.save(os.path.join(plot_to, "numeric_univariate.png"))

    # univariate distributions (counts) for the categorical variables

    categorical_plot = aly.dist(
        train_df, dtype="object", color="y"
    ).properties(
        title="Figure 2: Univariate distributions of categorical variables"
    )
    categorical_plot.save(
        os.path.join(plot_to, "categorical_univariate.png")
    )
    # Correlation Plot
    correlation_plot = aly.corr(bank_marketing_sample).properties(
        title="Figure 3: Correlation plot for numeric variables"
    )
    correlation_plot.save(
        os.path.join(plot_to, "correlation_plot.png")
    )
    print("Saved correlation plot")

    print("All tasks completed!")


if __name__ == '__main__':
    main()
