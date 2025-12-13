"""
This script contains a function to generate and save EDA visualizations.
"""
import os
import pandas as pd
import altair_ally as aly

def create_visualizations(train_df, full_df, plot_to):
    """
    Generates and saves all EDA visualizations.

    This function creates and saves three plots:
    1. Distribution of numeric variables.
    2. Distribution of categorical variables.
    3. Correlation matrix of numeric variables.

    Parameters
    ----------
    train_df : pd.DataFrame
        The training data subset.
    full_df : pd.DataFrame
        The complete dataset for the correlation plot.
    plot_to : str
        The directory path to save the plots.
    
    Raises
    ------
    TypeError
        If train_df or full_df are not pandas DataFrames.
    ValueError
        If DataFrames are empty or missing required columns.
    OSError
        If the output directory cannot be created or written to.
    KeyError
        If required column 'y' is missing from the DataFrame.
    """
    try:
        # Validate input types
        if not isinstance(train_df, pd.DataFrame):
            raise TypeError(f"train_df must be a pandas DataFrame, got {type(train_df).__name__}")
        if not isinstance(full_df, pd.DataFrame):
            raise TypeError(f"full_df must be a pandas DataFrame, got {type(full_df).__name__}")
        
        # Validate DataFrames are not empty
        if train_df.empty:
            raise ValueError("train_df cannot be empty")
        if full_df.empty:
            raise ValueError("full_df cannot be empty")
        
        # Validate required column exists
        if 'y' not in train_df.columns:
            raise KeyError("Column 'y' is required in train_df but was not found")
        
        # Validate output directory
        if not isinstance(plot_to, str):
            raise TypeError(f"plot_to must be a string, got {type(plot_to).__name__}")
        
        # Create output directory if it doesn't exist
        os.makedirs(plot_to, exist_ok=True)
        
        # Generate and save numeric variable distributions
        numeric_plot = aly.dist(train_df, color="y")
        numeric_plot.properties(
            title="Figure 1: Univariate distributions of numeric variables"
        ).save(os.path.join(plot_to, "numeric_univariate.png"))
        
        # Generate and save categorical variable distributions
        categorical_plot = aly.dist(
            train_df, dtype="object", color="y"
        ).properties(
            title="Figure 2: Univariate distributions of categorical variables"
        ).save(os.path.join(plot_to, "categorical_univariate.png"))
        
        # Generate and save correlation plot
        correlation_plot = aly.corr(full_df).properties(
            title="Figure 3: Correlation plot for numeric variables"
        ).save(os.path.join(plot_to, "correlation_plot.png"))
        
        print("Saved all plots successfully.")
        
    except (TypeError, ValueError, KeyError, OSError):
        print(f"Error in create_visualizations")
        raise
