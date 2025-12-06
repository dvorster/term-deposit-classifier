# download_data.py
# author: Tiffany Timbers
# date: 2023-11-27

import click
import os
from ucimlrepo import fetch_ucirepo 
import pandas


def read_uci_id(id, directory):
    """
    Read in a data set from the UCI Machine Learning repository using their API and save the 
    contents to a specified directory.

    Parameters:
    -----------
    id : int
        The ucimlrepo id of the dataset to read in. 
    directory : str
        The directory where the data set will be saved.

    Returns:
    --------
    None
    """

    # Check if directory exists, if not raise error
    if not os.path.isdir(directory):
        raise ValueError('The directory provided does not exist.')

    # Fetch the data from the UCI ML repo
    uci_data = fetch_ucirepo(id=id)

    # Create dataset and take a sample
    raw_uci_data=uci_data.data.features; raw_uci_data['y']=uci_data.data.targets 
    raw_uci_data_sample = raw_uci_data.sample(4000, random_state=522)

    # Create filename
    filename_data = "raw_data.csv"
    filename_data_sample = "raw_data_sample.csv"
    
    # Build path
    full_path_data = os.path.join(directory, filename_data)
    full_path_data_sample = os.path.join(directory, filename_data_sample)
    
    # write data to directory
    raw_uci_data.to_csv(full_path_data)
    raw_uci_data_sample.to_csv(full_path_data_sample)
    

@click.command()
@click.option('--id', type=str, help="id of dataset to be downloaded")
@click.option('--write_to', type=str, help="Path to directory where raw data will be written to")
def main(id, write_to):
    """Download data from UCI ML repo and save it."""
    try:
        read_uci_id(id, write_to)
    except:
        os.makedirs(write_to)
        read_uci_id(id, write_to)

if __name__ == '__main__':
    main()
