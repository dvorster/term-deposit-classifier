# download data code adapted from Tiffany A. Timbers, Joel Ostblom & Melissa Lee 2023/11/09's code
#Check for correct data file format:

import click
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.read_uci_id import read_uci_id

@click.command()
@click.option('--id', type=str, help="id of dataset to be downloaded")
@click.option('--write_to', type=str, help="Path to directory where raw data will be written to")
def main(id, write_to):
    """Download data from UCI ML repo and save it."""
    try:
        read_uci_id(id, write_to)
    except Exception as e:
        os.makedirs(write_to)
        read_uci_id(id, write_to)

if __name__ == '__main__':
    main()
