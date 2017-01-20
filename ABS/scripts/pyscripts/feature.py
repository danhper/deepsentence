# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pickle
import time

from gensim.models import Word2Vec
import numpy as np
import pandas as pd

import config
import dataset
import id2vector


parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_path', type=str)
parser.add_argument('--w2v_path', type=str)
parser.add_argument('--dictionary_path', type=str)
parser.add_argument('--save_dir', type=str)
args = parser.parse_args()

print('feature')

### dictionaryのload ###
token2id = dataset.load_dictionary(args.dictionary_path)
vocab_size = len(list(token2id.keys()))
id2token = {i:t for t, i in token2id.items()}
config.params.vocab_size = vocab_size
print('vocab size: ', vocab_size)

### word2vec ####
print('making id-vector dictionary')
w2v_model = Word2Vec.load_word2vec_format(args.w2v_path, binary=True)
id_vec_dic = id2vector.make_id_vector_dic(w2v_model, id2token, vocab_size)
del w2v_model

with open(args.batch_path, 'rb') as f_batch:
    list_batch = pickle.load(f_batch)


for batch in list_batch:
    batch = id2vector.convert_batch_plus_feature(batch, id_vec_dic, token2id)

