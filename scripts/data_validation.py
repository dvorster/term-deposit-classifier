# Script for validating data

import click
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.validate_data import validate_data

@click.command()
@click.option('--raw_data', type=str, help="Path to raw data")
def main(raw_data):
    validate_data(raw_data)

if __name__ == '__main__':
    main()


