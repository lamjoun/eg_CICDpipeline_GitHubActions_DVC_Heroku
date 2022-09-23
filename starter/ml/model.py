"""
Description:
    presents a library of function useful for manipulation \
    in connection with model

Creator: Udacity
Contributor: Rachid LAMJOUN

Date: 21 Sept., 2022

Version: 1.0
"""
#
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from .data import process_data
import pandas as pd
import joblib

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
features = ['age', 'workclass', 'fnlgt',
            'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex',
            'hours-per-week', 'native-country', 'salary']


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.
    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    #
    # Model Def
    model = RandomForestClassifier(n_estimators=100, random_state=10,
                                   n_jobs=-1)
    #
    # Train and save a model
    model.fit(X_train, y_train)
    #
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, \
    recall, and F1.
    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.
    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    #
    l_target = "salary"
    l_X_df = pd.DataFrame(X, columns=features[0:-1])
    l_X_df[l_target] = 0
    #
    loaded_encoder = joblib.load('models/encoder.joblib')
    loaded_lb = joblib.load('models/lb.joblib')
    #
    X_transf, y_transf, l_encoder, l_lb = \
        process_data(l_X_df, categorical_features=cat_features,
                     label=l_target, training=False, encoder=loaded_encoder,
                     lb=loaded_lb)
    #
    l_y_pred = model.predict(X_transf)
    return l_y_pred
