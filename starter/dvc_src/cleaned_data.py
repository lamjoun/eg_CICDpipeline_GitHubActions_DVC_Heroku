"""
Description: The second stage in dvc dag \
    clean data after loading

Author: Rachid LAMJOUN

Date: 21 Sept., 2022

Version: 1.0
"""
#
from ml.data import clean_data
import pandas as pd
import os
#
# export PYTHONPATH=starter   ==> to use ml.data

# loading data
root_dir = os.getcwd()
df = pd.read_csv(root_dir + '/data_dvc/original_data.csv')

# ***Cleaning  data...
data = clean_data(df)

# ***Saving df...
data.to_csv('data_dvc/cleaned_data.csv', index=False)
