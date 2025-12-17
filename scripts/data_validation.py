"""
Data validation script for bank marketing data.

This script uses the pandera package to validate
the data set, checking for correct column names, 
data types, and outliers or anaomalous values.

Author: Devon Vorster
Date: 2025-12-16
"""

import os
import sys
import click
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.validate_data import validate_data

@click.command()
@click.option('--raw_data', type=str, help="Path to raw data")
def main(raw_data):
    """
    This script validates the data, checking for: 
        - correct column names
        - correct data types in each column 
        - no outlier or anomalous values
    It does not change the data.


    Parameters:
    -----------
    raw_data : str
        path to the raw data CSV file

    Returns:
    --------
    None
    """
    validate_data(raw_data)

if __name__ == '__main__':
    main()


