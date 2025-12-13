import pytest
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.write_csv import write_csv

# Set up test id for valid and invalid id's
# Correct = 222, incorrect = 5
@pytest.fixture
def valid_id():
    return 222

@pytest.fixture
def invalid_id():
    return -1

    
# Test files setup

# setup empty directory for data files to be downloaded to 
if not os.path.exists('tests/test_data'):
    os.makedirs('tests/test_data')

# setup directory that contains a file for data files to be downloaded to
if not os.path.exists('tests/test_zip_data2'):
    os.makedirs('tests/test_zip_data2')
with open('tests/test_zip_data2/test4.txt', 'w') as file:
    pass  # The 'pass' statement does nothing, creating an empty file


# Tests

# Tests successfuly reads to data/raw folder, validate data exists and is what I want it to be

# Tests throws error for id is not an integer

# Tests throws an error for id that is not a valid repo

# Test throws an error for directory doesnt exist






test_files_txt_csv = ['test1.txt', 'test2.csv']
test_files_subdir = ['test1.txt', 'test2.csv', 'subdir/test3.txt']
test_files_2txt_csv = ['test1.txt', 'test2.csv', 'test4.txt']

# URL for Case 1 (zip file containing 'test1.txt' and 'test2.csv')
url_txt_csv_zip = 'https://github.com/ttimbers/breast_cancer_predictor_py/raw/main/tests/files_txt_csv.zip'

# URL for Case 2 ('test1.txt', test2.csv and 'subdir/test2.txt')
url_txt_subdir_zip = 'https://github.com/ttimbers/breast_cancer_predictor_py/raw/main/tests/files_txt_subdir.zip'

# URL for Case 3 (empty zip file)
url_empty_zip = 'https://github.com/ttimbers/breast_cancer_predictor_py/raw/main/tests/empty.zip'

# mock non-existing URL
@pytest.fixture
def mock_response():
    # Mock a response with a non-200 status code
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, 'https://example.com', status=404)
        yield

