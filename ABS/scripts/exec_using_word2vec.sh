#!/bin/bash

set -eu

REUTERS_PATHS_TEMPLATE='../../../scraper/output/reuters/\*/\*/\*.csv'
DATASET_DIR='../data/dataset'
FROM_DB_DIR='../data/raw_data/from_db'
DICTIONARY_DIR='../data/dictionary'

mkdir -p ${DATASET_DIR}
mkdir -p ${DICTIONARY_DIR}

# if [ -f ${DICTIONARY_DIR}/tokens.csv -a -f ${DICTIONARY_DIR}/alldata.dict -a -f ${DICTIONARY_DIR}/bow.pkl ]
# then
# 	echo 'skip make dictionary' 
# else
# 	python pyscripts/make_dictionary.py \
# 		   --reuters_paths_template ${REUTERS_PATHS_TEMPLATE} \
# 		   --from_db_path ${FROM_DB_DIR}/sources.pkl \
# 		   --save_dir ${DICTIONARY_DIR}
# fi

# if [ -f ${DATASET_DIR}/train.csv -a -f ${DATASET_DIR}/test.csv -a -f ${DATASET_DIR}/dictionary.pkl -a -f ${DATASET_DIR}/batch.pkl ]
# then
# 	echo 'skip making dataset'
# else
# 	python pyscripts/dataset.py \
# 		   --reuters_paths_template ${REUTERS_PATHS_TEMPLATE} \
# 		   --from_db_path ${FROM_DB_DIR}/sources.pkl \
# 		   --dictionary_dir ${DICTIONARY_DIR} \
# 		   --save_dir ${DATASET_DIR}  
# fi

W2V_PATH='../data/entity_vector/entity_vector.model.bin'

# MODEL_DIR='../result_using_word2vec/models'
# mkdir -p ${MODEL_DIR}

export CUDA_VISIBLE_DEVICES='1'
# python pyscripts/train_ABS_using_word2vec.py \
# 	   --batch_path ${DATASET_DIR}/batch.pkl \
# 	   --w2v_path ${W2V_PATH} \
# 	   --dictionary_path ${DATASET_DIR}/dictionary.pkl \
# 	   --save_dir ${MODEL_DIR}

MODEL_PLUS_DIR='../result_plus_using_word2vec/models'
mkdir -p ${MODEL_PLUS_DIR}
python pyscripts/train_ABS_plus_using_word2vec.py \
	   --batch_path ${DATASET_DIR}/batch.pkl \
	   --w2v_path ${W2V_PATH} \
	   --dictionary_path ${DATASET_DIR}/dictionary.pkl \
	   --save_dir ${MODEL_PLUS_DIR}
