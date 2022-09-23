"""
Description: The forth stage in dvc dag \
    train the rfc model and save the result module \

Author: Rachid LAMJOUN

Date: 21 Sept., 2022

Version: 1.0
"""
#
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model
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
root_dir = os.getcwd()
cat_features = config['data']['cat_features']
train = pd.read_csv(root_dir + '/data_dvc/train.csv')
test = pd.read_csv(root_dir + '/data_dvc/test.csv')
#
# *** load encoder and lb
encoder = joblib.load(root_dir+'/data_dvc/'+'encoder.sav')
lb = joblib.load(root_dir+'/data_dvc/'+'lb.sav')
#
# *** for train data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features,
    label="salary", training=False, encoder=encoder, lb=lb)

# *** for test data
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features,
    label="salary", training=False, encoder=encoder, lb=lb)

#
model = RandomForestClassifier(n_estimators=params['n_estimators'],
                               random_state=params['random_state'],
                               n_jobs=params['n_jobs'])

# Train a Random Forest (rf) model
model = train_model(X_train, y_train)
#
# *** Saving Random Forest model
filename_rf = '/data_dvc/' + 'rfc_model.sav'
joblib.dump(model, open(root_dir+filename_rf, 'wb'))
#
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(y_pred[0:10])
f1 = f1_score(y_test, y_pred)
print(f1)
