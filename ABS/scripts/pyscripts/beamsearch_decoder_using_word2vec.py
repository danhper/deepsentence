# -*- coding: utf-8 -*-

import os
import sys
import argparse
import math
import time

from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import tensorflow as tf

from models_no_embedding_layer import ABSmodel
import dataset
import config

import id2vector

MAX_OUTPUT_LENGTH = 15
BEAM_WIDTH = 3
INPUT_WEIGHT_BEFORE_SOFTMAX = 0.0

pd.set_option('display.width', 10000)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--model_path', type=str)
args = parser.parse_args()

# model_path = '../result_using_word2vec/models/epoch3-batch5000/model.ckpt'
# model_path = '../result_using_word2vec/models/epoch7-batch10000/model.ckpt'
model_path = '../result_using_word2vec/models/epoch15-batch10000/model.ckpt'

dataset_path = '../data/dataset/train.csv' #################################### test.csv
w2v_path = '../data/entity_vector/entity_vector.model.bin'

dictionary_path = '../data/dataset/dictionary.pkl'
token2id = dataset.load_dictionary(dictionary_path)
id2token = {i:t for t, i in token2id.items()}

symbol_ids = {'<S>': token2id['<S>'], '<EOS>': token2id['<EOS>']}
vocab_size = len(list(token2id.keys()))
config.params.vocab_size = vocab_size

print('loading dataset...')
dataset = dataset.str2list(dataset.load_dataset(dataset_path, 1, 100))

print('loading word2vec model...')
w2v_model = Word2Vec.load_word2vec_format(w2v_path, binary=True)
id_vec_dic = id2vector.make_id_vector_dic(w2v_model, id2token, vocab_size)
del w2v_model

print('setput graph')
sess = tf.Session()
if args.gpu:
    with tf.device('/gpu:%d'%args.gpu):
        model = ABSmodel(config.params)
        model.rebuild_forward_graph(sess, model_path)
else:
    model = ABSmodel(config.params)
    model.rebuild_forward_graph(sess, model_path)

# print(dataset)
start = time.time()
for row in dataset.iterrows():
    print('========================================================================')
    output = [np.array([token2id['<S>']]*config.params.window_size)]*BEAM_WIDTH
    output_prob = [1]*BEAM_WIDTH
    x = np.array(row[1]['x_labels']).astype(np.int32)
    t = np.array(row[1]['yc_and_t_labels']).astype(np.int32)
    x_words = ' '.join([id2token[a] for a in x])
    t_words = ' '.join([id2token[a] for a in t])
    
    x_weight = np.zeros(vocab_size)
    x_weight[x] = INPUT_WEIGHT_BEFORE_SOFTMAX

    print('x words: ', x_words)
    print('t words: ', t_words)
    x_vector = np.array([id_vec_dic[a] for a in x])[np.newaxis, :, :]
    print('---------------------------------')
    for i in range(MAX_OUTPUT_LENGTH):
        temp_output = np.array([])
        temp_output_prob = np.array([])
        for j, o in enumerate(output):
            ### model decoder ###
            y_c = o[-3: ]
            y_c_words = ' '.join([id2token[a] for a in y_c])
            # print(y_c_words)
            y_c_vector = np.array([id_vec_dic[a] for a in y_c])[np.newaxis, :, :].reshape(1, -1)
            prob = np.squeeze(model.decode(sess, x_vector, y_c_vector, x_weight), 0)
            #print('prob', prob.shape)
            
            ### beam search  ###
            prob_id = np.sort(prob)[-1::-1][: BEAM_WIDTH] * output_prob[j]
            #print('prob_id', prob_id.shape)
            candidate_ids = np.array([np.r_[output[j], np.array([word_id])] for word_id in np.argsort(prob)[-1::-1][: BEAM_WIDTH]])
            #print('candidate_ids', candidate_ids.shape)

            #print('temp_output', temp_output.shape)
            #print('candidate_ids', candidate_ids.shape)
            #print('prob_id', prob_id.shape)
            
            temp_output_prob = np.r_[temp_output_prob, prob_id]
            #print('temp_output_prob', temp_output_prob.shape)

            if temp_output.size == 0:
                temp_output = candidate_ids
            else:
                temp_output = np.r_[temp_output, candidate_ids]
        
            if i == 0:
                break
        
        choice = np.argsort(temp_output_prob)[-1::-1][: BEAM_WIDTH]
        output = temp_output[choice]
        output_prob = temp_output_prob[choice]

        # print('------------')
        # for l, o in enumerate(output):
        #     o_words = ' '.join([id2token[a] for a in o])
        #     print('candidate%d: %s'%(l, o_words))
        
        # print(output.shape)
        # print(output_prob.shape)
        
    o = output[0]
    # print('=================')
    # print('t words: ', t_words)
    o_words = ' '.join([id2token[a] for a in o])
    print('output word: ', o_words)
print(time.time()-start)
