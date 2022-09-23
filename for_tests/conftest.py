"""
Description: Contain Pytest fixtures for the test of projects functions \
    concretely these contexts are used by the test library test_functions

Author: Rachid LAMJOUN
Date: 19th September, 2022

Version: 1.0
"""
#
# Import libraries
import yaml
import pytest
import starter.ml.data
from sklearn.model_selection import train_test_split


@pytest.fixture
def _config_object():
    """
    To fetch data configuration

    Parameters
    ----------
    Returns
    -------
        obj_config : dictionary
            data of configuration
    """
    #
    with open('config.yaml') as file:
        obj_config = yaml.safe_load(file)
    return obj_config


@pytest.fixture
def _load_data(_config_object):
    """
    Give information in order to load data

    Parameters
    ----------
        _config_object:
            the previous context for data config
    Returns
    -------
        l_path_file_csv : string
            path the file
        l_path_file_csv : string
            repository address which contains the dvc pointer \
            to real data in S3 aws bucket
        l_col_num: int
            used to test the expected number of columns
    """
    #
    l_config = _config_object
    l_path_file_csv = l_config['data']['path_csv']
    l_repo_url = l_config['data']['repo_url']
    l_col_num = l_config['data']['col_num']
    #
    return l_path_file_csv, l_repo_url, l_col_num


@pytest.fixture
def _clean_data(_load_data):
    """
    load data to prepare the representing df for cleaning

    Parameters
    ----------
        _load_data
            the previous context necessary for loading data
    Returns
    -------
        l_df : Dataframe
            cleaned data for precessing necessary for processing after...
    """
    #
    l_path_file_csv, l_repo_url, l_col_num = _load_data
    l_path_file_csv = l_path_file_csv
    l_df = starter.ml.data.load_data(l_path_file_csv, l_repo_url)
    #
    return l_df


@pytest.fixture
def _process_data(_clean_data):
    """
    Return dataframe for test after cleaning

    Parameters
    ----------
        _clean_data
            the previous context necessary for loading data
    Returns
    -------
        l_df2 : Dataframe
            cleaned data for precessing necessary for processing after...
    """
    #
    l_df = _clean_data
    l_df = starter.ml.data.clean_data(l_df)
    _, l_df2 = train_test_split(l_df, test_size=0.30)
    #
    return l_df2
