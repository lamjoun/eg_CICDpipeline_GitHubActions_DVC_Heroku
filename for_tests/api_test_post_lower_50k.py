"""
Description:
    Test post() function on url: \
    url = 'http://127.0.0.1:8000/test_salary_prediction'
    - Test the result and the status code
    - Test case result <=50K

Creator: Rachid LAMJOUN

Date: 25 Sept., 2022

Version: 1.0
"""
import json
import logging
import requests
#
logging.basicConfig(level=logging.INFO)
#
url = 'http://127.0.0.1:8000/test_salary_prediction'
# url = 'https://census-case-via-heroku.herokuapp.com/test_salary_prediction'

input_data_test = {
    "age": 18,
    "work_class": "Private",
    "fn_lgt": 192409,
    "education": "Some-college",
    "marital_status": "Never-married",
    "occupation": "Sales",
    "relationship": "Own-child",
    "race": "White",
    "sex": "Female",
    "hours_per_week": 20,
    "native_country": "United-States"
}

input_json = json.dumps(input_data_test)
response = requests.post(url, data=input_json)
#
res_code = response.status_code
response_txt = response.text
str_sup_50k = "\"<=50K\""
#
try:
    assert res_code == 200
    assert response_txt == str_sup_50k
except AssertionError as err:
    logging.error("Test KO ...")
    raise err
#
logging.error("\n===== Test OK =====")
logging.info("Status code=" + str(res_code))
logging.info("Response: " + response_txt + "\n")
