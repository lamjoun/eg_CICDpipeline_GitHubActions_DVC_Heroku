"""
Description: The Third stage in dvc dag \
    process data to prepare operation for \
    train operation after

Author: Rachid LAMJOUN

Date: 21 Sept., 2022

Version: 1.0
"""
#
from sklearn.model_selection import train_test_split
from ml.data import process_data
import pandas as pd
import joblib
import yaml
import os


with open('config.yaml') as file:
    config = yaml.safe_load(file)
#
with open('params.yaml') as file:
    params = yaml.safe_load(file)
#
cat_features = config['data']['cat_features']
#
root_dir = os.getcwd()
df = pd.read_csv(root_dir + '/data_dvc/cleaned_data.csv')
#
test_size = params['test_size']
train, test = train_test_split(df, test_size=test_size)
#
# ***Saving output files...
# col_x = list(df.columns[0:-1])
# col_y =[df.columns[-1]]
#
train.to_csv('data_dvc/train.csv', index=False)
test.to_csv('data_dvc/test.csv', index=False)
#
# *** for train data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
#
# *** Saving Encoder model
filename_encoder = '/data_dvc/' + 'encoder.sav'
joblib.dump(encoder, open(root_dir+filename_encoder, 'wb'))

# *** Saving lb model
filename_lb = '/data_dvc/' + 'lb.sav'
joblib.dump(lb, open(root_dir+filename_lb, 'wb'))
