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


def sample_dataset(data, k, balance=False) -> pd.DataFrame:
    """Randomly sample a given dataset

    :param data: DataFrame of the data (two columns: review, sentiment)
    :param k: desired total number of data points (across all classes)
    :param balance: to balance class ratio or not
    :return: a DataFrame of the sampled data
    """
    g = data.groupby('sentiment')

    if balance:
        n = min(g.size().min(), k // len(g))
        sample_data = g.apply(lambda x: x.sample(n=n)).reset_index(drop=True)
    else:
        frac = float(k) / float(data.shape[0])
        sample_data = g.apply(lambda x: x.sample(frac=frac)).reset_index(drop=True)

    return sample_data
