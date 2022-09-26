"""
Description:
    Test get() function on url: \
    'http://127.0.0.1:8000/test_salary_prediction'
    - Test the result and the status code

Creator: Rachid LAMJOUN

Date: 25 Sept., 2022

Version: 1.0
"""
#
import logging
import requests
#
logging.basicConfig(level=logging.INFO)
#
url = 'http://127.0.0.1:8000'

response = requests.get(url, data=None)
#
res_code = response.status_code
dict_resp = response.json()
message_resp = dict_resp["message"]
message = "Hi, here function to predict Salary.."
#
try:
    assert res_code == 200
    assert message_resp == message
except AssertionError as err:
    logging.error("Test KO ...")
    logging.error(message_resp)
    raise err
#
logging.error("\n===== Test OK =====")
logging.info("Status code=" + str(res_code))
logging.info("Response: " + message_resp + "\n")
