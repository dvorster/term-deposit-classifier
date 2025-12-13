import pytest
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.read_uci_id import read_uci_id

# Set up test id for valid and invalid id's
# Correct = 222, incorrect = 5
@pytest.fixture
def valid_id():
    return 222

@pytest.fixture
def invalid_id():
    return -1

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
def read_uci_success(valid_id, temp_directory):
    read_uci_id(id=valid_id, temp_directory)
    
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
    assert loaded_df_smaple.columns.tolist() == col_names
    assert len(loaded_df) == sample_length


# Tests throws error for id is not an integer

# Tests throws an error for id that is not a valid repo

# Test throws an error for directory doesnt exist

