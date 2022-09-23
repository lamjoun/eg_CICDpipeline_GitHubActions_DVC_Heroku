"""
Description: test if the data is reachable for reading. \
        The data is stored in an aws S3 bucket. we use \
        dvc to store this data

Author: Rachid LAMJOUN

Date: 21 Sept., 2022

Version: 1.0
"""

import dvc.api
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logging.info('Hi..')

with dvc.api.open(
    'data/census.csv',
    repo='https://github.com/lamjoun/eg_CICDpipeline_GitHubActions_DVC_Heroku'
     ) as fd:
    # print(fd.readline())
    df = pd.read_csv(fd)        # pandas.Series.str.replace(' ', '')
# print(df.head(3))
logging.info(df.head(3))

for col in df.columns:
    if df[col].dtype == object:
        df[col].str.strip()
#
df.columns = df.columns.str.replace(' ', '')

logging.info("Columns list: ")
logging.info(df.columns)
