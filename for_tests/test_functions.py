"""
Description: Contain function tests used by pytest \

Author: Rachid LAMJOUN
Date: 21th September, 2022

Version: 1.0
"""
#
import os
import main
import joblib
import logging
import warnings
import collections
import pandas as pd
import starter.ml.data
import starter.ml.model
import pandas.core.frame
from random import randint
from sklearn.model_selection import train_test_split

#
warnings.filterwarnings("ignore")
#
logging.basicConfig(level=logging.INFO)


#


def test_load_data(_load_data):
    """
    test data import - exactly the load_data() function of \
    the starter.ml.data file
    """
    l_path, l_url_repo, l_col_num = _load_data
    try:
        l_df = starter.ml.data.load_data(l_path, l_url_repo)
    except Exception as err:
        logging.info(
            "problem reading data csv file !!! ")
        raise err
    assert type(l_df) is pandas.core.frame.DataFrame
    assert len(l_df) > 0
    assert len(l_df.columns) == l_col_num


def test_clean_data(_clean_data, _config_object):
    """
    # test the output dataframe after the operation of cleaning \
    by tarter.ml.data.clean_data() function
    """
    l_df = _clean_data
    l_df = starter.ml.data.clean_data(l_df)
    l_columns = l_df.columns
    #
    ll = []
    for elm in list(l_df.salary.values):
        ll.append(elm.strip())
    #
    assert set(l_columns) == \
           set(_config_object['data']['columns_set_after_clean'])
    assert type(l_df) is pandas.core.frame.DataFrame
    assert len(l_df) > 0
    assert set(l_df.isnull().sum()) == {0}
    assert set(ll) == \
           set(_config_object['data']['salary_column_values'])


def test_process_data(_process_data, _config_object):
    """
    # test the output dataframe after the operation of cleaning \
    by tarter.ml.data.clean_data() function
    """
    #
    l_df = _process_data
    l_cat_features = list(_config_object['data']['cat_features'])
    #
    l_columns = l_cat_features + ["salary"]
    #
    l_X, l_y, l_encoder, l_lb = \
        starter.ml.data.process_data(l_df[l_columns],
                                     categorical_features=l_cat_features,
                                     label="salary",
                                     training=True)
    #
    l_X_inv = l_encoder.inverse_transform(l_X)
    l_df_cat_val = l_df[l_cat_features].values
    idx = randint(0, l_X_inv.shape[0])
    ll_l_X_inv = list(l_X_inv[idx, :])
    ll_l_df_cat_val = list(l_df_cat_val[idx, :])
    #
    ll_l_y_inv = list(l_lb.inverse_transform(l_y))
    ll_l_df_sal_val = list(l_df['salary'].values)
    #
    assert len(l_df) == len(l_X)
    assert len(l_X) == len(l_y)
    #
    assert collections.Counter(ll_l_X_inv) == \
           collections.Counter(ll_l_df_cat_val)
    assert collections.Counter(ll_l_y_inv) == \
           collections.Counter(ll_l_df_sal_val)


def test_saved_models(_config_object):
    """
    test if the models: rfc_model.joblib, encoder.joblib and \
    lb.joblib have been stored properly
    """
    l_path = _config_object['model']['models_path']
    root_dir = os.getcwd()
    #
    logging.error(root_dir)
    assert os.path.isfile(root_dir + l_path + "rfc_model.joblib")
    assert os.path.isfile(root_dir + l_path + "encoder.joblib")
    assert os.path.isfile(root_dir + l_path + "lb.joblib")


def test_performance(_process_data, _config_object):
    """
    test if the rfc model have the acceptable metrics: \
    precision, recall and fbeta. the minimums declared \
    in config file
    """
    #
    l_test_size = _config_object['model']['test_size']
    l_df = _process_data
    _, l_test = train_test_split(l_df, test_size=l_test_size)
    #
    cols = l_test.columns[0:-1]
    l_X_arr = l_test[cols].values
    l_y = l_test['salary'].values
    #
    # ***** Prediction....
    root = os.getcwd()
    l_path = root + _config_object['model']['models_path']
    #
    l_loaded_model = joblib.load(l_path + 'rfc_model.joblib')
    loaded_lb = joblib.load(l_path + 'lb.joblib')
    #
    l_y_prd = starter.ml.model.inference(l_loaded_model, l_X_arr)
    l_y = loaded_lb.transform(l_y)
    #
    l_precision, l_recall, l_fbeta = \
        starter.ml.model.compute_model_metrics(l_y, l_y_prd)

    l_precision_min = _config_object['model']['metrics']['precision_min']
    l_recall_min = _config_object['model']['metrics']['recall_min']
    l_fbeta_min = _config_object['model']['metrics']['fbeta_min']
    #
    l_condition = (l_precision > l_precision_min) and \
                  (l_recall > l_recall_min) and \
                  (l_fbeta > l_fbeta_min)
    #
    logging.error("=======o======")
    precision_str = " ==> " + str(l_precision) + "   " + str(l_precision_min)
    recall_str = " ==> " + str(l_recall) + "   " + str(l_recall_min)
    fbeta_str = " ==> " + str(l_fbeta) + "   " + str(l_fbeta_min)
    logging.error(precision_str + recall_str + fbeta_str)
    #
    assert l_condition


def test_fct_post_for_prediction(_config_object):
    """
    To get closer to the final case we test the function, \
    used by the post call in main.py file
    """
    #
    root_dir = os.getcwd()
    df_tmp_00 = pd.read_csv(root_dir + '/data/test_df_prd_0.csv')
    df_tmp_11 = pd.read_csv(root_dir + '/data/test_df_prd_1.csv')
    #
    df_tmp_00.rename(columns={'workclass': 'work_class'},
                     inplace=True)
    df_tmp_00.rename(columns={'fnlgt': 'fn_lgt'}, inplace=True)
    df_tmp_00.rename(columns={'marital-status': 'marital_status'},
                     inplace=True)
    df_tmp_00.rename(columns={'hours-per-week': 'hours_per_week'},
                     inplace=True)
    df_tmp_00.rename(columns={'native-country': 'native_country'},
                     inplace=True)
    #
    df_tmp_11.rename(columns={'workclass': 'work_class'},
                     inplace=True)
    df_tmp_11.rename(columns={'fnlgt': 'fn_lgt'}, inplace=True)
    df_tmp_11.rename(columns={'marital-status': 'marital_status'},
                     inplace=True)
    df_tmp_11.rename(columns={'hours-per-week': 'hours_per_week'},
                     inplace=True)
    df_tmp_11.rename(columns={'native-country': 'native_country'},
                     inplace=True)
    #
    l_len_0 = len(df_tmp_00)
    l_len_1 = len(df_tmp_11)
    #
    df_tmp_0_restCol = df_tmp_00[df_tmp_00.columns[0:-2]]
    df_tmp_1_restCol = df_tmp_11[df_tmp_11.columns[0:-2]]
    #
    df_tmp_0_lst = df_tmp_0_restCol.to_dict('records')
    df_tmp_1_lst = df_tmp_1_restCol.to_dict('records')
    #
    ok_sup_50K = False
    ok_inf_50K = False
    #
    for idx in range(0, l_len_1):
        data_for_prediction = df_tmp_1_lst[idx]
        input_data = main.ModelInput(**data_for_prediction)
        l_prediction = main.test_salary_prediction(input_data)
        if l_prediction == '>50K':
            ok_sup_50K = True
            break
    #
    for idx in range(0, l_len_0):
        data_for_prediction = df_tmp_0_lst[idx]
        input_data = main.ModelInput(**data_for_prediction)
        l_prediction = main.test_salary_prediction(input_data)
        if l_prediction == '<=50K':
            ok_inf_50K = True
            break

    assert ok_sup_50K and ok_inf_50K
