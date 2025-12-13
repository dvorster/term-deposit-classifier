"""
Module for testing read_uci_id.py.

This module contains tests to ensure functionality of 
read_uci_id.py. read_uci_id.py should fetch data from
the uci ml repository using their built in api, and wite
the data to a provided dictionary for further analysis. 
It should raise errors if an invalid id key is passed or if
a non-existent data frame is passed.

Author: Devon Vorster
Date: 2025-12-13
"""


import pytest
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.read_uci_id import read_uci_id

# Set up test id for valid and invalid id's
# Correct = 222, incorrect = 5
@pytest.fixture
def valid_id():
    return '222'

@pytest.fixture
def invalid_id():
    return '-1'

@pytest.fixture
def temp_directory(tmp_path):
    return tmp_path

@pytest.fixture
def col_names():
    return ['age',
 'job',
 'marital',
 'education',
 'default',
 'balance',
 'housing',
 'loan',
 'contact',
 'day_of_week',
 'month',
 'duration',
 'campaign',
 'pdays',
 'previous',
 'poutcome',
 'y']

@pytest.fixture
def length():
    return 45211

@pytest.fixture
def sample_length():
    return 4000

    
# # Test files setup

# # setup empty directory for data files to be downloaded to 
# if not os.path.exists('tests/test_data'):
#     os.makedirs('tests/test_data')

# # setup directory that already contains data files
# if not os.path.exists('tests/test_data2'):
#     os.makedirs('tests/test_data2')
# with open('tests/test_data2/raw_data.csv', 'w') as file:
#     pass  # The 'pass' statement does nothing, creating an empty file
# with open('tests/test_data2/raw_data_sample.csv', 'w') as file:
#     pass  # The 'pass' statement does nothing, creating an empty file

# Tests

# Tests successfuly reads to data/raw folder, validate data exists and is what I want it to be
def test_read_uci_id_success(valid_id, temp_directory, col_names, length, sample_length):
    read_uci_id(valid_id, temp_directory)
    
    # Check that the file exists
    file_path = os.path.join(temp_directory, 'raw_data.csv')
    assert os.path.isfile(file_path)

    # Check that the file exists
    file_path_sample = os.path.join(temp_directory, 'raw_data_sample.csv')
    assert os.path.isfile(file_path_sample)

    # Validate the contents of the CSV files
    loaded_df = pd.read_csv(file_path)
    assert loaded_df.columns.tolist() == col_names
    assert len(loaded_df) == length

    loaded_df_sample = pd.read_csv(file_path_sample)
    assert loaded_df_sample.columns.tolist() == col_names
    assert len(loaded_df_sample) == sample_length


# Tests throws error for id which is not an integer
def test_read_uci_id_non_int_id(temp_directory):
    non_int_id = '22l'
    
    with pytest.raises(ValueError, match="id must be provided as an integer in string format"):
        read_uci_id(non_int_id, temp_directory)

# Tests throws an error for id that is not a valid repo
def test_read_uci_id_invalid_id(temp_directory):
    invalid_int_id = '-1'
    
    with pytest.raises(FileNotFoundError, match=f'Repository matching {invalid_int_id} does not exist'):
        read_uci_id(invalid_int_id, temp_directory)

# Test throws an error for directory doesnt exist
def test_read_uci_id_directory_doesnt_exist(temp_directory):
    valid_id = '1'
    invalid_directory = '/invalid_directory'
    
    with pytest.raises(FileNotFoundError, match='The directory provided does not exist.'):
        read_uci_id(valid_id, invalid_directory)

