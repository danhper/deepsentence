# -*- coding: utf-8 -*-

import os
import sys
import math
import random
import time
import logging

import numpy as np
import tensorflow as tf

class NNLMmodel(object):

    def __init__(self, params):
        self.params = params

    def encoder(self):
        pass

    def build_train_graph(self):

        batch_size = self.params.batch_size
        window_size = self.params.window_size
        vocab_size = self.params.vocab_size
        embedding_size = self.params.embedding_size
        hidden_size = self.params.hidden_size
        
        ### word embedding layer ###
        # E: y_c(CV) -> tilde_y_c(CD)
        # C: window_size
        # V: vocab_size
        # D: embedding_size
        self.y_c = tf.placeholder(tf.int32, shape=[batch_size, window_size]) # window_sizeを可変にする必要があるだろね
        self.E = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0)) # 乱数の設定を参照
        self.tilde_y_c = tf.reshape(tf.nn.embedding_lookup(self.E, self.y_c), shape=[batch_size, window_size*embedding_size])

        ### fully connected layer ###
        # U: tilde_y_c(CD) -> h(H)
        # H: hidden_size
        self.U_w = tf.Variable(tf.truncated_normal((window_size*embedding_size, hidden_size),
                                                   stddev=1.0/math.sqrt(hidden_size))) # 乱数の設定を参照
        self.U_b = tf.Variable(tf.zeros([hidden_size]))
        self.h = tf.nn.tanh(tf.matmul(self.tilde_y_c, self.U_w) + self.U_b)

        ### fully connected layer ####
        # V_: h(H) -> p(y_i+1|x, y_c;theta)(V)
        self.V_w = tf.Variable(tf.truncated_normal((hidden_size, vocab_size),
                                                   stddev=1.0/math.sqrt(vocab_size))) # 乱数の設定を参照
        self.V_b = tf.Variable(tf.zeros([vocab_size]))
        self.prob_from_h = tf.matmul(self.h, self.V_w) + self.V_b

        ### encoder ###
        # M: length_x
        self.x = tf.placeholder(tf.int32, shape=[batch_size, None])
        
        # if enc_type == 'Bag-of-Words':
        #     self.bag_of_words_encoder()
        # elif enc_type == 'Attention-Based':
        #     self.attention_based_encoder()

        self.encoder()
        
        ### fully connceted layer ###
        # W: enc(H) ->  p(y_i+1|x, y_c;theta)(V)
        self.W_w = tf.Variable(tf.truncated_normal((hidden_size, vocab_size),
                                                   stddev=1.0/math.sqrt(vocab_size))) # 乱数の設定を参照
        self.W_b = tf.Variable(tf.zeros([vocab_size]))
        self.prob_from_enc = tf.matmul(self.h, self.W_w) + self.W_b

        ###   ###
        # p(y_i+1|x, y_c;theta)(V) ~ exp(Vh + Wenc(x, y_c))
        self.prob = tf.nn.softmax(self.prob_from_h + self.prob_from_enc)

        ### for training ###
        self.t = tf.cast(tf.placeholder(tf.int32, shape=[batch_size, vocab_size]), tf.float32)
        self.cross_entropy = -tf.reduce_sum(self.t * tf.log(self.prob)) 
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.prob, 1), tf.argmax(self.t, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
    
        
class BOWmodel(NNLMmodel):
        
    def encoder(self):
        batch_size = self.params.batch_size
        vocab_size = self.params.vocab_size
        hidden_size = self.params.hidden_size

        self.F = tf.Variable(tf.random_uniform([vocab_size, hidden_size], -1.0, 1.0)) # 乱数の設定を参照
        self.tilde_x = tf.cast(tf.nn.embedding_lookup(self.F, self.x), tf.float32)
        #print(self.tilde_x.get_shape()) # (batch_size, None(x_length), hidden_size)
        
        self.p = tf.placeholder(tf.float32, shape=[batch_size, None]) # (batch_size, None(x_length))

        for i in range(batch_size):
            # print(tf.slice(self.p, [i, 0], [1, -1]).get_shape()) # (1, hidden_size)
            # print(tf.slice(self.tilde_x, [i, 0, 0], [1, -1, -1]).get_shape()) # (1, None(x_length), hidden_size)
            # print(tf.reshape(tf.slice(self.tilde_x, [i, 0, 0], [1, -1, -1]), [-1, hidden_size]).get_shape()) # (None(x_length), hidden_size)
            temp = tf.matmul(tf.slice(self.p, [i, 0], [1, -1]), tf.reshape(tf.slice(self.tilde_x, [i, 0, 0], [1, -1, -1]), [-1, hidden_size])) 
            # print(temp.get_shape()) # (1, hidden_size)
            if i == 0:
                self.enc = temp 
            else:
                self.enc = tf.concat(0, [self.enc, temp])

        # print(self.enc.get_shape()) # (batch_size, hidden_size)
    def train(self, session, x, p, y_c, t):
        feed_dict = {self.x: x, self.p: p, self.y_c: y_c, self.t: t}
        self.train_step.run(session=session, feed_dict=feed_dict)
        return self.accuracy.eval(session=session, feed_dict=feed_dict)

class ABSmodel(NNLMmodel):
                
    def encoder(self):

        batch_size = self.params.batch_size
        window_size = self.params.window_size
        vocab_size = self.params.vocab_size
        embedding_size = self.params.embedding_size
        hidden_size = self.params.hidden_size
        smoothing_window_size = self.params.smoothing_window_size

        self.G = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0)) # 乱数の設定を参照
        self.tilde_y_c_dash = tf.reshape(tf.nn.embedding_lookup(self.G, self.y_c), shape=[batch_size, window_size*embedding_size])
        self.F = tf.Variable(tf.random_uniform([vocab_size, hidden_size], -1.0, 1.0)) # 乱数の設定を参照
        self.tilde_x = tf.cast(tf.nn.embedding_lookup(self.F, self.x), tf.float32)
        self.P = tf.Variable(tf.truncated_normal((window_size*embedding_size, hidden_size),
                                                 stddev=1.0/math.sqrt(hidden_size))) # 乱数の設定を参照
        self.p_ = tf.matmul(self.tilde_y_c_dash, self.P)
        self.smoothed_x = tf.cast(tf.placeholder(tf.int32, shape=[batch_size, None, vocab_size]), tf.float32) / (2*smoothing_window_size+1)
        # print(self.p_.get_shape()) # (batch_size, hidden_size)
        # print(self.tilde_x.get_shape()) # (batch_size, None(hidden_size, hidden_size))
        
        for i in range(batch_size):
            # print(tf.slice(self.p_, [i, 0], [1, -1]).get_shape()) # (1, hidden_size)
            # print(tf.slice(self.tilde_x, [i, 0, 0], [1, -1, -1]).get_shape()) # (1, None(x_length), hidden_size)
            # print(tf.reshape(tf.slice(self.tilde_x, [i, 0, 0], [1, -1, -1]), [-1, hidden_size]).get_shape()) # (None(x_length), hidden_size)
            # print(tf.reshape(tf.slice(self.smoothed_x, [i, 0, 0], [1, -1, -1]), [-1, vocab_size]).get_shape()) # (None(x_length), vocab_size)
            # print(self.F.get_shape()) # (vocab_size, hidden_size)
            
            p = tf.matmul(tf.slice(self.p_, [i, 0], [1, -1]), tf.transpose(tf.reshape(tf.slice(self.tilde_x, [i, 0, 0], [1, -1, -1]), [-1, hidden_size])))
            x_bar = tf.matmul(tf.reshape(tf.slice(self.smoothed_x, [i, 0, 0], [1, -1, -1]), [-1, vocab_size]), self.F)
            temp = tf.matmul(p, x_bar)

            # print(p.get_shape())
            # print(x_bar.get_shape())
            # print(temp.get_shape())

            if i == 0:
                self.enc = temp
            else:
                self.enc = tf.concat(0, [self.enc, temp])
                
    def train(self, session, x, smoothed_x, y_c, t):
        feed_dict = {self.x: x, self.smoothed_x: smoothed_x, self.y_c: y_c, self.t: t}
        self.train_step.run(session=session, feed_dict=feed_dict)
        return self.accuracy.eval(session=session, feed_dict=feed_dict)
        
if __name__ == '__main__':

    import config

    x = np.array([[3, 443, 545, 1080, 793, 0, 0],
                  [435, 413, 1380, 0, 0, 0, 0],
                  [49, 233, 366, 1356, 45, 0, 0],
                  [1035, 3123, 150, 0, 0, 0, 0],
                  [31, 3, 51, 3106, 313, 36, 1126]])
    p = np.array([[1/5.0, 1/5.0, 1/5.0, 1/5.0, 1/5.0, 0.0,   0.0],
                  [1/3.0, 1/3.0, 1/3.0, 0.0,   0.0,   0.0,   0.0],
                  [1/5.0, 1/5.0, 1/5.0, 1/5.0, 1/5.0, 0.0,   0.0],
                  [1/3.0, 1/3.0, 1/3.0, 0.0,   0.0,   0.0,   0.0],
                  [1/7.0, 1/7.0, 1/7.0, 1/7.0, 1/7.0, 1/7.0, 1/7.0]]).astype(np.float32)
    y_c = np.array([[305, 3667, 14],
                    [3, 1678, 6517],
                    [962, 36677, 0],
                    [60, 126, 1126],
                    [16671, 1, 367]])
    t = np.array([3663, 5, 1360, 9378, 777])
    t_onehot = np.zeros((config.params.batch_size, config.params.vocab_size))
    t_onehot[np.arange(config.params.batch_size), t] = 1
    

    print('x.shape: ', x.shape)
    print('p.shape: ', p.shape)
    print('y_c.shape: ', y_c.shape)
    
    model = BOWmodel(config.params)
    model.build_graph()
    output = model.train(x, p, y_c, t_onehot)
    
   
