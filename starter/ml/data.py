"""
Description:
    presents a library of data processing functions. \
    Functions used by several tariffs in particular by train_model.py

Creator: Udacity
Contributor: Rachid LAMJOUN

Date: 21 Sept., 2022

Version: 1.0
"""
#
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import pandas as pd
import numpy as np
import dvc.api


def load_data(i_path_file_csv, i_repo_url):
    """
        fetch data from S3 Aws-bucket, data in csv file \
        and present the data as df (Dataframe)
    Inputs
    ------
        i_path_file_csv: string
            representing the path the data to fetch
        i_repo_url: string
            representing the repository address which \
            contain the pointer dvc to S3 bucket contains \
            the real data
    Returns
    -------
        l_df : pd.DataFrame
            representing the fetched data
    """
    with dvc.api.open(
            i_path_file_csv,
            repo=i_repo_url
    ) as fd:
        l_df = pd.read_csv(fd)
    #
    return l_df


def clean_data(i_df):
    """
        Clean Dataframe from special characters, duplicates and \
        drop three columns without distribution:
        capital-gain, capital-loss and education-num
    Inputs
    ------
        i_df : pd.DataFrame
            Representing the data to clean-up
    Returns
    -------
        l_df : pd.DataFrame
            The df cleaned
    """

    for col in i_df.columns:
        if i_df[col].dtype == object:
            i_df[col].str.strip()
    #
    i_df.columns = i_df.columns.str.replace(' ', '')

    """
  for i in range(len(i_df.columns)):
    l_new_col = i_df.columns[i].strip()
    i_df.rename(columns = {i_df.columns[i]:l_new_col}, inplace = True)
  """
    #
    # print(i_df.columns)
    #
    l_df = i_df.copy().drop_duplicates()
    #
    l_df.replace({'?': None}, inplace=True)
    l_df.dropna(inplace=True)
    #
    l_df.drop("capital-gain", axis=1, inplace=True)  # No distribution
    l_df.drop("capital-loss", axis=1, inplace=True)  # No distribution
    #
    l_df.drop("education-num", axis=1, inplace=True)
    # correlated with education

    df_obj = l_df.select_dtypes(['object'])
    l_df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    return l_df


def process_data(
        X, categorical_features=[], label=None, training=True,
        encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.
    Processes the data using one hot encoding for the categorical features \
    and a label binarizer for the labels. This can be used in either \
    training or inference/validation.
    Note: depending on the type of model used, you may want to add in \
    functionality that scales the continuous data.
    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in
        `categorical_features` categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array
        will be returned for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise
        returns the encoder passed in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns
        the binarizer passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
