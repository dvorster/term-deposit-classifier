# download data code adapted from Tiffany A. Timbers, Joel Ostblom & Melissa Lee 2023/11/09's code
#Check for correct data file format:

"""
Script to download dataset from UCI ML repository.

This script uses the ucimlrepo API to download data
using an integer key value realting to a dataset.
The dataset is then read into the raw data folder.

Author: Devon Vorster
Date: 2025-12-16
"""

import click
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.read_uci_id import read_uci_id

@click.command()
@click.option('--id', type=str, help="id of dataset to be downloaded")
@click.option('--write_to', type=str, help="Path to directory where raw data will be written to")
def main(id, write_to):
    """
    Read in a data set from the UCI Machine Learning 
    repository using their API and save the 
    contents to a specified directory.

    Parameters:
    -----------
    id : int
        The ucimlrepo id of the dataset to read in. 
    write_to : str
        The path to the directory where the data set will be saved.

    Raises:
    -------
    ValueError
        If the id value is not an integer.
    FileNotFoundError
        If the id does not point to a valid ucimlrepo 
        repository, or if directory is not a valid directory
    
    Returns:
    --------
    None
    """
    try:
        read_uci_id(id, write_to)
    except Exception as e:
        os.makedirs(write_to)
        read_uci_id(id, write_to)

if __name__ == '__main__':
    main()
