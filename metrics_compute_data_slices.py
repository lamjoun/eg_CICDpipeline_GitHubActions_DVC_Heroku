"""
Description:
    Give model metrics for each data slice having a particular value \
     for each categorical feature. \
     An output file is given by feature.

Creator: Rachid LAMJOUN

Date: 21 Sept., 2022

Version: 1.0
"""
#

import os
import yaml
import joblib
import logging
import warnings
from sklearn.metrics import classification_report

from starter.ml.data import load_data, clean_data
from starter.ml.model import inference, compute_model_metrics, cat_features
#
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
#
# val_col = 'HS-grad'


def slice_metrics(data, i_col, val_col, i_name_file,
                  i_loaded_model, i_loaded_lb):
    """
        Generate file with model metrics for each data slice having \
        a particular value for feature passed as parameter
    Parameters
    ----------
    data: Dataframe
        represent all data of file
    i_col: String
        feature concerned by slicing
    val_col: String
        value of feature which define one slice
    i_name_file: String
        output file
    i_loaded_model: sklearn.ensemble._forest.RandomForestClassifier
        rfc model which used for prediction
    i_loaded_lb: sklearn.preprocessing._label.LabelBinarizer'
        LabelBinarizer model used for numerical conversion

    Returns
    -------
        output file for categorical feature passed as a parameter

    """
    df_t = data[data[i_col] == val_col]
    df_t.head(5)
    #
    cols = df_t.columns[0:-1]
    X_arr = df_t[cols].values
    y_df_t = df_t['salary'].values
    #
    y_pred_df_t = inference(i_loaded_model, X_arr)

    y_df_t = i_loaded_lb.transform(y_df_t)

    (o_precision2, o_recall2, o_fbeta2) = \
        compute_model_metrics(y_df_t, y_pred_df_t)
    # print('o_precision2=',o_precision2)
    # print('o_recall2=',o_recall2)
    # print('o_fbeta2=',o_fbeta2)
    with open(i_name_file, "a") as f:
        f.write("#--------------------#" + "\n")
        f.write('******' + val_col + '\n')
        f.write('#--------------------#\n')
        f.write(classification_report(y_df_t, y_pred_df_t))
        #
        f.write("\n" + 'o_precision2=' + str(o_precision2) + "\n")
        f.write('o_recall2=' + str(o_recall2) + "\n")
        f.write('o_fbeta2=' + str(o_fbeta2) + "\n")
        f.write('#---------------------------------------#\n')


with open('config.yaml') as file:
    l_config = yaml.safe_load(file)
path_file_csv = l_config['data']['path_csv']
repo_url = l_config['data']['repo_url']
l_models_path = l_config['model']['models_path']
l_test_size = l_config['model']['test_size']
#

# ***Loading data...
logging.info("\n")
logging.info("*****O***** Loading data...")
df = load_data(path_file_csv, repo_url)

# ***Cleaning  data...
logging.info("*****O***** Cleaning data...")
data = clean_data(df)

root = os.getcwd() + l_models_path
loaded_model = joblib.load(root+'rfc_model.joblib')
#
loaded_encoder = joblib.load(root+'encoder.joblib')
loaded_lb = joblib.load(root+'lb.joblib')

# print('loaded_model',type(loaded_model))
# print('loaded_lb',type(loaded_lb))

if not os.path.exists('./metrics_output_files'):
    os.makedirs('./metrics_output_files')

output_path = "./metrics_output_files/"
# cat_features1=['education', 'race']
cat_features1 = cat_features
for col in cat_features1:
    file_out = output_path + "slice_" + "output_" + col + ".txt"
    try:
        os.remove(file_out)
    except OSError:
        pass
    for val in list(set(data[col].values)):
        slice_metrics(data, col, val, file_out, loaded_model, loaded_lb)
