"""
Module for reading data fromt the UCI ML repository using their API.

This module contains functionality to read in data from the UCI ML
repository using their built in API. It ensures data is read in 
correctly and saved as a csv file in the correct data/ subfolder.

Author: Devon Vorster
Date: 2025-12-13
"""

from ucimlrepo import fetch_ucirepo 
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.write_csv import write_csv

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

    Raises:
    -------
    ValueError
        If the id value is not an integer.
    FileNotFoundError
        If the id does not point to a valid ucimlrepo repository, or if directory is not a valid directory
    
    Returns:
    --------
    None
    """
    # Cast id to int
    try :
        id=int(id)
    except Exception as e:
        raise ValueError("id must be provided as an integer in string format")
     
    # Check if directory exists, if not raise error
    if not os.path.isdir(directory):
        raise FileNotFoundError('The directory provided does not exist.')

    # Fetch the data from the UCI ML repo
    try :
        uci_data = fetch_ucirepo(id=id)
    except Exception as e:
        raise FileNotFoundError(f'Repository matching {id} does not exist')

    # Check data from ucimlrepo is a pandas dataframe
    if not isinstance(uci_data.data.features, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame") 
    if not isinstance(uci_data.data.targets, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")  

    # Combine features and targets
    raw_uci_data=uci_data.data.features; raw_uci_data['y']=uci_data.data.targets

    # Check Data is not empty
    if raw_uci_data.empty:
        raise ValueError("DataFrame must contain observations.")

    # Take random sample of data
    raw_uci_data_sample = raw_uci_data.sample(4000, random_state=522)

    # Check Random Sample of Data is not empty
    if raw_uci_data_sample.empty:
        raise ValueError("DataFrame must contain observations.")

    # write to CSV
    write_csv(raw_uci_data, directory, "raw_data.csv")
    write_csv(raw_uci_data_sample, directory, "raw_data_sample.csv")

    