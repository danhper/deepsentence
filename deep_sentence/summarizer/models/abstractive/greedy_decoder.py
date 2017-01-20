# -*- coding: utf-8 -*-

import os
import sys
import argparse
import math
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from models import ABSmodel
import dataset
import config

MAX_OUTPUT_LENGTH = 20

pd.set_option('display.width', 10000)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--model_path', type=str)
args = parser.parse_args()

model_path = '../result/models/epoch3/model.ckpt'
dataset_path = '../data/dataset/train.csv' #################################### test.csv

dictionary_path = '../data/dataset/dictionary.pkl'
token2id = dataset.load_dictionary(dictionary_path)
id2token = {i:t for t, i in token2id.items()}

symbol_ids = {'<S>': token2id['<S>'], '<EOS>': token2id['<EOS>']}
vocab_size = len(list(token2id.keys()))
config.params.vocab_size = vocab_size

dataset = dataset.str2list(dataset.load_dataset(dataset_path, 1, 100))

sess = tf.Session()
if args.gpu:
    with tf.device('/gpu:%d'%args.gpu):
        model = ABSmodel(config.params)
        model.rebuild_forward_graph(sess, model_path)
else:
    model = ABSmodel(config.params)
    model.rebuild_forward_graph(sess, model_path)

# print(dataset)

for row in dataset.iterrows():
    print('=======================================================')
    output = [token2id['<S>']]
    output_prob = [-1]
    x = np.array(row[1]['x_labels']).astype(np.int32)
    t = np.array(row[1]['yc_and_t_labels']).astype(np.int32)
    x_words = ' '.join([id2token[a] for a in x])
    t_words = ' '.join([id2token[a] for a in t])
    x = x[np.newaxis, :]
    print('x words: ', x_words)
    print('t words: ', t_words)
    y_c = np.array([token2id['<S>']]*config.params.window_size).astype(np.int32)
    y_c = y_c[np.newaxis, :]
    print('---------------------------------')
    while output[-1] != token2id['<EOS>']:
        y_c_words = ' '.join([id2token[a] for a in np.squeeze(y_c, 0)])
        pred, pred_prob, _ = model.decode(sess, x, y_c)
        output.append(pred)
        output_prob.append(pred_prob)
        y_c = np.c_[y_c[:, 1:], np.array([pred])[np.newaxis, :]]
        
        if len(output) > MAX_OUTPUT_LENGTH:
            output.append(token2id['<EOS>'])
            break
    output_words = ' '.join([id2token[a] for a in output])
    print('output: ', output_words)
    
