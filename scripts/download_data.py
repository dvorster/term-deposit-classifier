# download_data.py
# author: Tiffany Timbers
# date: 2023-11-27

import click
import os
from ucimlrepo import fetch_ucirepo 


@click.command()
@click.option('--id', type=str, help="id of dataset to be downloaded")
@click.option('--write_to', type=str, help="Path to directory where raw data will be written to")
def main(id, write_to):
    """
    Read in a dataset with id from the UCI machine learning repo, using the ucimlrepo API

    Parameters:
    ----------
    id : int
        The id number of the dataset to read.
    write_ot : str
        The directory where the data will be written to.

    Returns:
    -------
    None
    """

    # Fetch the data from the UCI ML repo
    bank_marketing = fetch_ucirepo(id=id)

    # Get the data as pandas dataframes
    X = bank_marketing.data.features 
    y = bank_marketing.data.targets 

    # create complete dataset
    bank_marketing_data =X; bank_marketing_data['y'] = y
    bank_marketing_data.to_csv('data/bank_marketing.csv')

    # take a sample from teh data set
    bank_marketing_sample = bank_marketing_data.sample(4000, random_state=522)
    bank_marketing_sample.to_csv('data/bank_marketing_small.csv')

    # ADDsection to use os to read to data safely!! and to check that data is read etc!!