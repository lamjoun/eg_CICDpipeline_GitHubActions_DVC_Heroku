# -*- coding: utf-8 -*-
"""
Created 09 09 2022
@author: R. LAMJOUN
"""

import os
import json
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
#
from starter.ml.model import inference

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelInput(BaseModel):
    age: int
    work_class: str
    fn_lgt: int
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    hours_per_week: int
    native_country: str


list_clumns = ['age', 'work_class', 'fn_lgt', 'education', 'marital_status',
               'occupation', 'relationship', 'race', 'sex', 'hours_per_week',
               'native_country']


@app.get("/")
async def get_items():
    """
    GET on the root giving a welcome message.
    """
    return {"message": "Hi, here function to predict Salary.."}


@app.post('/test_salary_prediction')
def test_salary_prediction(input_parameters: ModelInput):
    #
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    # print("----------input_dictionary-----------")
    # print(input_dictionary)
    #
    input_list = []
    for elm in list_clumns:
        input_list.append(input_dictionary[elm])

    X_arr = np.empty((1, len(input_list)), dtype=object)
    for i in range(0, len(input_list)):
        X_arr[0, i] = input_list[i]
    #
    # l_target = "salary"
    # l_X_df = pd.DataFrame(X_arr, columns=list_clumns)
    # l_X_df[l_target] = 0
    #
    # ***** Prediction....
    root = os.getcwd()
    #
    model = joblib.load(root + '/models/rfc_model.joblib')
    #
    # print("\n\n\n_arr")
    # print(X_arr)
    #
    prediction = inference(model, X_arr)

    model_lb = joblib.load(root + '/models/lb.joblib')

    return str(model_lb.inverse_transform(prediction)[0]).strip()


"""
    if prediction[0] == 0:
        return "<50K"
    elif prediction[0] == 1:
        return ">50K"
    else:
        return "no decision!!"
"""
