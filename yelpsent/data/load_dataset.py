"""
load_dataset
"""

from pathlib import Path

import pandas as pd


def load_dataset(train, test) -> (pd.DataFrame, pd.DataFrame):
    """Load the Yelp review data

    :param train: path to the training JSON file
    :param test: path to the testing JSON file
    :return: DataFrames of the training and testing data
    """
    train_path = Path(train)
    test_path = Path(test)

    if train_path.suffix != '.json' or test_path.suffix != '.json':
        raise TypeError("Please provide JSON files!")

    if not train_path.exists():
        raise FileNotFoundError("{0} does not exist!".format(train_path.resolve()))

    if not test_path.exists():
        raise FileNotFoundError("{0} does not exist!".format(test_path.resolve()))

    return pd.read_json(train_path), pd.read_json(test_path)
