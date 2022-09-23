"""
Description: The first stage in dvc dag \
    load data from storage S3 bucket

Author: Rachid LAMJOUN

Date: 21 Sept., 2022

Version: 1.0
"""
#
from ml.data import load_data
import yaml
#
# export PYTHONPATH=starter   ==> to use ml.data

with open('config.yaml') as file:
    config = yaml.safe_load(file)
#
path_file_csv = config['data']['path_csv']
repo_url = config['data']['repo_url']

# ***Loading data...
df = load_data(path_file_csv, repo_url)

# ***Saving df...
df.to_csv('data_dvc/original_data.csv', index=False)
