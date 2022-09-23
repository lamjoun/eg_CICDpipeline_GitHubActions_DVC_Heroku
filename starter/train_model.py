"""
Description:
    Train machine learning model.
    The model is a RandomForestClassifier

Creator: Udacity
Contributor: Rachid LAMJOUN

Date: 21 Sept., 2022

Version: 1.0
"""
#
# import for libraries
import os
import yaml
import joblib
import logging
import warnings
from ml.model import train_model
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from ml.data import load_data, clean_data, process_data
#
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
#
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
#
# root_dir = '/content/'
# root_dir = env['GITHUB_WORKSPACE']
# root_dir = '.'
#
root_dir = os.getcwd()
#
# path_file = root_dir + '/data/census.csv'
#
#
with open('config.yaml') as file:
    l_config = yaml.safe_load(file)
path_file_csv = l_config['data']['path_csv']
repo_url = l_config['data']['repo_url']
l_models_path = l_config['model']['models_path']
l_test_size = l_config['model']['test_size']
l_folder_models = l_config['model']['folder_models']
#

# ***Loading data...
logging.info("\n")
logging.info("*****O***** Loading data...")
df = load_data(path_file_csv, repo_url)

# ***Cleaning  data...
logging.info("*****O***** Cleaning data...")
data = clean_data(df)

# Optional enhancement, use K-fold cross validation instead
# of a train-test split.
train, test = train_test_split(data, test_size=l_test_size)
#
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.

# Train a Random Forest (rf) model
logging.info("*****O***** training rfc (random forest classifier) model...")
model = train_model(X_train, y_train)
#
if not os.path.isdir(l_folder_models):
    os.mkdir(l_folder_models)
#
# *** Saving Random Forest model
logging.info("*****O***** Saving the rfc Model...")
filename_rf = l_models_path + 'rfc_model.joblib'
joblib.dump(model, open(root_dir+filename_rf, 'wb'))

# *** Saving Encoder model
logging.info("*****O***** Saving encoder Model...")
filename_encoder = l_models_path + 'encoder.joblib'
joblib.dump(encoder, open(root_dir+filename_encoder, 'wb'))

# *** Saving lb model
logging.info("*****O***** Saving lb Model...\n")
filename_lb = l_models_path + 'lb.joblib'
joblib.dump(lb, open(root_dir+filename_lb, 'wb'))

# ============ o ============
# Process the test data with the process_data function
X_test, y_test, encoder, lb = process_data(test,
                                           categorical_features=cat_features,
                                           label="salary", training=False,
                                           encoder=encoder, lb=lb)
# ***** Prediction...
root = os.getcwd() + '/models/'
loaded_model = joblib.load(root+'rfc_model.joblib')
#
loaded_encoder = joblib.load(root+'encoder.joblib')
loaded_lb = joblib.load(root+'lb.joblib')
#
y_prd = loaded_model.predict(X_test)
logging.info(classification_report(y_test, y_prd))
logging.info("\n")
logging.info("*****O***** Prediction for X_test...\n")
logging.info("For test---> Prediction for some entries of X_test")
logging.info(y_prd[50:60])
logging.info("\n")

#

df_test_tmp = test.copy(deep=False)
df_test_tmp['prediction'] = 0

ll = []
for i in range(0, len(test)):
    if y_prd[i] == 1:
        ll.append(test.iloc[i])
        df_test_tmp['prediction'][i] = 1
print(ll[1:3])

df_tmp_0 = df_test_tmp[df_test_tmp['prediction'] ==
                       0][df_test_tmp['salary'] == '<=50K']
df_tmp_1 = df_test_tmp[df_test_tmp['prediction'] ==
                       1][df_test_tmp['salary'] == '>50K']

root_dir = os.getcwd()
# file contains data of test df which prediction = 0
df_tmp_0.to_csv(root_dir + '/data/test_df_prd_0.csv', index=False)
#
# file contains data of test df which prediction = 1
df_tmp_1.to_csv(root_dir + '/data/test_df_prd_1.csv', index=False)
