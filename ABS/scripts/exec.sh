#!/bin/bash

set -eu

REUTERS_PATHS_TEMPLATE='../../../scraper/output/reuters/\*/\*/\*.csv'
REUTERS_DATASET_DIR='../data/dataset/reuters'
mkdir -p ${REUTERS_DATASET_DIR}

if [ -f ${REUTERS_DATASET_DIR}/train.pkl -a -f ${REUTERS_DATASET_DIR}/test.pkl ]
then
	:
else
	python pyscripts/reuters_dataset.py \
		   --data_paths_template ${REUTERS_PATHS_TEMPLATE} \
		   --save_dir ${REUTERS_DATASET_DIR}  
fi


MODEL_PATH='../result/model/reuters.ckpt'
mkdir -p ../result/model
python pyscripts/train_ABS.py \
	   --dataset_path ${REUTERS_DATASET_DIR}/train.pkl \
	   --save_path ${MODEL_PATH}

python pyscripts/decoder_ABS.py \
	   --dataset_path ${REUTERS_DATASET_PATH}/test.pkl \
	   --model_path ${MODEL_PATH} 
