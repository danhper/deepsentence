#!/bin/bashs

n_rows=None
output_path="../dataset/dataset_row${n_rows}.pkl"

python pyscripts/fetch_rawdata.py \
	   ${output_path}

