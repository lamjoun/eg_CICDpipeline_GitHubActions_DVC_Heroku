#!/bin/bash
#./script_dvc_stages.sh
# By execution it define dvc.yaml file --> file necessary fo stages
dvc stage add -n original_data \
                -d starter/dvc_src/original_data.py \
                -o data_dvc/original_data.csv \
                python starter/dvc_src/original_data.py

dvc stage add -n cleaned_data \
                -d starter/dvc_src/cleaned_data.py -d data_dvc/census.csv \
                -o data_dvc/cleaned_data.csv \
                python starter/dvc_src/cleaned_data.py

dvc stage add -n OneHot_encoder \
                -d starter/dvc_src/OneHot_encoder.py -d data_dvc/cleaned_data.csv \
                -p test_size \
                -o data_dvc/train.csv -o data_dvc/test.csv \
                -o data_dvc/encoder.sav -o data_dvc/lb.sav \
                python starter/dvc_src/OneHot_encoder.py

dvc stage add -n model \
                -d starter/dvc_src/model.py -d data_dvc/train.csv -d data_dvc/test.csv \
                -d data_dvc/encoder.sav -d data_dvc/lb.sav \
                -p n_estimators -p random_state -p n_jobs \
                -o data_dvc/rfc_model.sav  \
                python starter/dvc_src/model.py

