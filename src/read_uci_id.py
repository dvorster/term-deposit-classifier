from ucimlrepo import fetch_ucirepo 
import pandas
import write_csv.py

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
    except ():
        raise ValueError("id must be provided as an integer in string format")
     
    # Check if directory exists, if not raise error
    if not os.path.isdir(directory):
        raise FileNotFoundError('The directory provided does not exist.')

    # Fetch the data from the UCI ML repo
    try :
        uci_data = fetch_ucirepo(id=id)
    except():
        raise FileNotFoundError(f'Repository matching {id} does not exist')

    # Check data from ucimlrepo is a pandas dataframe
    if not isinstance(uci_data.data.features, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame") 
    if not isinstance(raw_uci_data_sample, pd.DataFrame):
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

    # Read to CSV
    write_csv(raw_uci_data, directory, "raw_data.csv")
    write_csv(raw_uci_data_sample, directory, "raw_data_sample.csv")

    