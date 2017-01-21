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

from models_no_embedding_layer import ABS_plus_model
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

# model_path = '../result_plus_using_word2vec/models/epoch1-batch5000/model.ckpt'
model_path = '../result_plus_using_word2vec/models/epoch1-batch39244/model.ckpt'

dataset_path = '../data/dataset/test.csv' #################################### test.csv
w2v_path = '../data/entity_vector/entity_vector.model.bin'

dictionary_path = '../data/dataset/dictionary.pkl'
token2id = dataset.load_dictionary(dictionary_path)
id2token = {i:t for t, i in token2id.items()}

symbol_ids = {'<S>': token2id['<S>'], '<EOS>': token2id['<EOS>']}
vocab_size = len(list(token2id.keys()))
config.params.vocab_size = vocab_size

print('loading dataset...')
dataset_content = dataset.str2list(dataset.load_dataset(dataset_path, 1, 100))

print('loading word2vec model...')
w2v_model = Word2Vec.load_word2vec_format(w2v_path, binary=True)
id_vec_dic = id2vector.make_id_vector_dic(w2v_model, id2token, vocab_size)
del w2v_model

print('setput graph')
sess = tf.Session()
if args.gpu:
    with tf.device('/gpu:%d'%args.gpu):
        model = ABS_plus_model(config.params)
        model.rebuild_forward_graph(sess, model_path)
else:
    model = ABS_plus_model(config.params)
    model.rebuild_forward_graph(sess, model_path)

alpha_ph = tf.placeholder(tf.float32, shape=[5])
sess.run(model.alpha.assign(alpha_ph),
         feed_dict={alpha_ph: np.array([1, 0.1, 0.1, 0.1, 0.1]).astype(np.float32)})
    
# print(dataset)
start = time.time()
for row in dataset_content.iterrows():
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
            y_c_vector = np.array([id_vec_dic[a] for a in y_c])[np.newaxis, :, :].reshape(1, -1)
            
            temp_x = np.r_[[token2id['<S>']]*config.params.window_size, x]
            reference_x = np.copy(temp_x)
            uni_id = np.unique(np.delete(reference_x, np.where((reference_x==token2id['<S>'])|(reference_x==token2id['<EOS>']))))
            
            reference_x = temp_x[: -1]
            bi_id = temp_x[np.where(reference_x==y_c[-1])[0]+1]
            bi_id = np.unique(np.delete(bi_id, np.where((bi_id==token2id['<S>'])|(bi_id==token2id['<EOS>']))))
        
            reference_x = temp_x[: -2]
            tri_point = np.where(np.r_[np.array([False]), (reference_x==y_c[-2])[:-1]] & (reference_x==y_c[-1]))[0]+1
            tri_id = temp_x[tri_point]
            tri_id = np.unique(np.delete(tri_id, np.where((tri_id==token2id['<S>'])|(tri_id==token2id['<EOS>']))))

            try:
                reordering_id = temp_x[np.arange(np.where(temp_x==y_c[-1])[0][0])]
                reordering_id = np.unique(np.delete(reordering_id, np.where((reordering_id==token2id['<S>'])|(reordering_id==token2id['<EOS>']))))
            except IndexError:
                reordering_id = np.array([])

            uni_vocab_vector = np.zeros(config.params.vocab_size)
            if len(uni_id) > 0:
                uni_vocab_vector[uni_id] = 1
            
            bi_vocab_vector = np.zeros(config.params.vocab_size)
            if len(bi_id) > 0:
                bi_vocab_vector[bi_id] = 1
            
            tri_vocab_vector = np.zeros(config.params.vocab_size)
            if len(tri_id) > 0:
                tri_vocab_vector[tri_id] = 1
            
            reordering_vocab_vector = np.zeros(config.params.vocab_size)
            if len(reordering_id) > 0:
                reordering_vocab_vector[reordering_id] = 1
            
            features = np.c_[uni_vocab_vector, bi_vocab_vector, tri_vocab_vector, reordering_vocab_vector][np.newaxis, :, :]

            prob = np.squeeze(model.decode(sess, x_vector, y_c_vector, x_weight, features), 0)

            ### beam search  ###
            prob_id = np.sort(prob)[-1::-1][: BEAM_WIDTH] * output_prob[j]

            candidate_ids = np.array([np.r_[output[j], np.array([word_id])] for word_id in np.argsort(prob)[-1::-1][: BEAM_WIDTH]])
            
            temp_output_prob = np.r_[temp_output_prob, prob_id]

            if temp_output.size == 0:
                temp_output = candidate_ids
            else:
                temp_output = np.r_[temp_output, candidate_ids]
        
            if i == 0:
                break
        
        choice = np.argsort(temp_output_prob)[-1::-1][: BEAM_WIDTH]
        output = temp_output[choice]
        output_prob = temp_output_prob[choice]

      
    o = output[0]

    o_words = ' '.join([id2token[a] for a in o])
    print('output word: ', o_words)
print(time.time()-start)
