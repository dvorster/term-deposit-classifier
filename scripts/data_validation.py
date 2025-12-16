"""
Data validation script for bank marketing data.

This script uses the pandera package to validate
the data set, checking for correct column names, 
data types, and outliers or anaomalous values.

Author: Devon Vorster
Date: 2025-12-16
"""

import click
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.validate_data import validate_data

@click.command()
@click.option('--raw_data', type=str, help="Path to raw data")
def main(raw_data):
    "validate data set using panderas functions"
    validate_data(raw_data)

if __name__ == '__main__':
    main()


