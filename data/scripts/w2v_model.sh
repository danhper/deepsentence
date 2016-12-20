#!/bin/bash

size=5
window=5
min_count=5

model_dir="../w2v_model/size${size}_window${window}_min${min_count}/"
model_path="${model_dir}/model.pkl"
mkdir -p ${model_dir}

python pyscripts/w2v_model.py \
	   --size ${size} \
	   --window ${window} \
	   --min_count ${min_count} \
	   ${model_path}
