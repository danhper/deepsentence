#!/bin/bashs

n_rows=100
output_path="../dataset/dataset_row${n_rows}.pkl"

python pyscripts/fetch_rawdata.py \
	   --n_rows ${n_rows} \
	   ${output_path}

