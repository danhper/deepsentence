#!/bin/bash

set -eu

REUTERS_PATHS_TEMPLATE='../../../scraper/output/reuters/\*/\*/\*.csv'
REUTERS_DATASET_DIR='../data/dataset/reuters'
mkdir -p ${REUTERS_DATASET_DIR}

if [ -f ${REUTERS_DATASET_DIR}/train.csv -a -f ${REUTERS_DATASET_DIR}/test.csv -a -f ${REUTERS_DATASET_DIR}/dictionary.pkl ]
then
	:
else
	python pyscripts/reuters_dataset.py \
		   --data_paths_template ${REUTERS_PATHS_TEMPLATE} \
		   --save_dir ${REUTERS_DATASET_DIR}  
fi


MODEL_DIR='../result/models/reuters'
mkdir -p ${MODEL_DIR}
python pyscripts/train_ABS.py \
	   --dataset_path ${REUTERS_DATASET_DIR}/train.csv \
	   --dictionary_path ${REUTERS_DATASET_DIR}/dictionary.pkl \
	   --save_dir ${MODEL_DIR}

# python pyscripts/decoder_ABS.py \
# 	   --dataset_path ${REUTERS_DATASET_DIR}/test.pkl \
# 	   --model_dir ${MODEL_DIR} 
