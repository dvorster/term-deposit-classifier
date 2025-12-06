# validate data code adapted from Tiffany A. Timbers, Joel Ostblom & Melissa Lee 2023/11/09's code
#Check for correct data file format:

import click
import pandas as pd
import pandera.pandas as pa

def assert_csv_format(file_path):
    assert file_path.endswith(".csv"), "Error: File must be a CSV."
    return True

@click.command()
@click.option('--raw-data', type=str, help="Path to raw data")
def main(raw_data):
    """
    This script validates the data, checking for correct column names, correct data types in each column, no outlier or anomalous values. It does not change the data.
    """
    
    marketing_sample = pd.read_csv(raw_data)


    #Validate Data:
    schema = pa.DataFrameSchema(
        {
            "age": pa.Column(int, pa.Check.between(18, 91), nullable=True),
            "job": pa.Column(object, pa.Check.isin(['technician', 'blue-collar', 'admin.', 'entrepreneur',
           'management', 'self-employed', 'retired', 'services', 'unemployed',
           'housemaid', 'student']), nullable=True),
            "marital": pa.Column(object, pa.Check.isin(['married', 'single', 'divorced']), nullable=True),
            "education": pa.Column(object, pa.Check.isin(['tertiary', 'secondary', 'primary']), nullable=True), 
            "default": pa.Column(object, pa.Check.isin(['no', 'yes']), nullable=True),
            "balance": pa.Column(int, pa.Check.between(-10000000, 10000000), nullable=True),
            "housing": pa.Column(object, pa.Check.isin(['yes', 'no']), nullable=True),
            "loan": pa.Column(object, pa.Check.isin(['no', 'yes']), nullable=True),
            "contact": pa.Column(object, pa.Check.isin(['cellular', 'telephone']), nullable=True),
            "day_of_week": pa.Column(int, pa.Check.between(1, 31), nullable=True),
            "month": pa.Column(object, pa.Check.isin(['feb', 'nov', 'jul', 'may', 'aug', 'jun', 'apr', 'mar', 'jan',
           'oct', 'sep', 'dec']), nullable=True),
            "duration": pa.Column(int, pa.Check.between(0, 4000), nullable=True),
            "campaign": pa.Column(int, pa.Check.between(0, 40), nullable=True),
            "pdays": pa.Column(int, pa.Check.between(-1, 800), nullable=True), 
            "previous": pa.Column(int, pa.Check.between(0, 30), nullable=True),
            "poutcome": pa.Column(object, pa.Check.isin(['failure', 'success', 'other']), nullable=True),
            "y": pa.Column(object, pa.Check.isin(['no', 'yes']), nullable=True),
        },
        #check for no duplicate observations, no empty observations and missingness not beyond expected threshold:
        checks=[
            pa.Check(lambda df: ~df.duplicated().any(), error="Duplicate rows found."),
            pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found."),
            pa.Check(lambda df: (df.isna().mean() < 0.10).any(), error="Missing values is above threshold.")
        ]
    )

    assert_csv_format(raw_data)
    schema.validate(marketing_sample, lazy=True)

if __name__ == '__main__':
    main()


