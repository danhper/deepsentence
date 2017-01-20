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
        variable_init = self.params.variable_init
        
        ### word embedding layer ###
        # E: y_c(CV) -> tilde_y_c(CD)
        # C: window_size
        # V: vocab_size
        # D: embedding_size
        self.y_c = tf.placeholder(tf.int32, shape=[batch_size, window_size]) 
        self.E = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -variable_init, variable_init), name='E') 
        self.tilde_y_c = tf.reshape(tf.nn.embedding_lookup(self.E, self.y_c), shape=[batch_size, window_size*embedding_size])

        ### fully connected layer ###
        # U: tilde_y_c(CD) -> h(H)
        # H: hidden_size
        self.U_w = tf.Variable(tf.truncated_normal((window_size*embedding_size, hidden_size),
                                                   stddev=variable_init/math.sqrt(hidden_size)), name='U_w') 
        self.U_b = tf.Variable(tf.zeros([hidden_size]), name='U_b')
        self.h = tf.nn.tanh(tf.matmul(self.tilde_y_c, self.U_w) + self.U_b)

        ### fully connected layer ####
        # V_: h(H) -> p(y_i+1|x, y_c;theta)(V)
        self.V_w = tf.Variable(tf.truncated_normal((hidden_size, vocab_size),
                                                   stddev=variable_init/math.sqrt(vocab_size)), name='V_w') 
        self.V_b = tf.Variable(tf.zeros([vocab_size]), name='V_b')
        self.prob_from_h = tf.matmul(self.h, self.V_w) + self.V_b

        ### encoder ###
        # M: length_x
        self.x = tf.placeholder(tf.int32, shape=[batch_size, None])

        self.encoder()
        
        ### fully connceted layer ###
        # W: enc(H) ->  p(y_i+1|x, y_c;theta)(V)
        self.W_w = tf.Variable(tf.truncated_normal((hidden_size, vocab_size),
                                                   stddev=variable_init/math.sqrt(vocab_size)), name='W_w')
        self.W_b = tf.Variable(tf.zeros([vocab_size]), name='W_b')
        self.prob_from_enc = tf.matmul(self.enc, self.W_w) + self.W_b

        ###   ###
        # p(y_i+1|x, y_c;theta)(V) ~ exp(Vh + Wenc(x, y_c))
        self.prob = tf.nn.softmax(self.prob_from_h + self.prob_from_enc)

        # ### for training ###
        self.t = tf.cast(tf.placeholder(tf.int32, shape=[batch_size, vocab_size]), tf.float32)
        self.cross_entropy = -tf.reduce_sum(self.t * tf.log(self.prob)) 
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.prob, 1), tf.argmax(self.t, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
    
        
class ABSmodel(NNLMmodel):
                
    def encoder(self):

        batch_size = self.params.batch_size
        window_size = self.params.window_size
        vocab_size = self.params.vocab_size
        embedding_size = self.params.embedding_size
        hidden_size = self.params.hidden_size
        smoothing_window_size = self.params.smoothing_window_size
        variable_init = self.params.variable_init
        
        self.G = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -variable_init, variable_init), name='G') 
        self.tilde_y_c_dash = tf.reshape(tf.nn.embedding_lookup(self.G, self.y_c), shape=[batch_size, window_size*embedding_size])

        self.F = tf.Variable(tf.random_uniform([vocab_size, hidden_size], variable_init, variable_init), name='F') 
        self.tilde_x = tf.cast(tf.nn.embedding_lookup(self.F, self.x), tf.float32)

        self.P = tf.Variable(tf.truncated_normal((window_size*embedding_size, hidden_size),
                                                 stddev=variable_init/math.sqrt(hidden_size)), name='P')

        self.p_ = tf.matmul(self.tilde_y_c_dash, self.P)

        self.tilde_x = tf.transpose(self.tilde_x, [1, 0, 2]) # (?, batch_size, hidden_size)

        initializer = [tf.reshape(tf.slice(self.tilde_x, [0, 0, 0], [1, -1, -1]), [batch_size, hidden_size])] * (smoothing_window_size+1) # 前window_size分は元のやつ、残り一つ分が欲しいやつ
        x_bar_forward = tf.scan(lambda a, x: a[1: -1] + [x] + [sum(a[: -1] + [x])], elems=self.tilde_x, initializer=initializer)[-1]
        
        reverse_tilde_x = tf.reverse(self.tilde_x, [True, False, False]) # reverseの扱い注意
        
        initializer = [tf.reshape(tf.slice(reverse_tilde_x, [0, 0, 0], [1, -1, -1]), [batch_size, hidden_size])] * (smoothing_window_size+1)
        x_bar_backward = tf.reverse(tf.scan(lambda a, x: a[1: -1] + [x] + [sum(a[: -1] + [x])], elems=reverse_tilde_x, initializer=initializer)[-1], [True, False, False])

        self.x_bar = tf.transpose((x_bar_forward + x_bar_backward - self.tilde_x), [1, 0, 2]) / (2*smoothing_window_size+1)
        
        self.tilde_x = tf.transpose(self.tilde_x, [1, 0, 2]) # x_barを計算した後

        self.enc = tf.reshape(tf.scan(lambda a, x: tf.matmul(tf.nn.softmax(tf.matmul(tf.expand_dims(x[0], 0), x[1])), x[2]), #(1, ?) (?, hidden_size)
                                      elems=(self.p_, tf.transpose(self.tilde_x, [0, 2, 1]), self.x_bar),
                                      initializer=tf.zeros([1, hidden_size])), [batch_size, hidden_size])

                
    def train(self, session, x, y_c, t):
        feed_dict = {self.x: x, self.y_c: y_c, self.t: t}
        self.train_step.run(session=session, feed_dict=feed_dict)

    
    def rebuild_graph(self, sess, model_path):

        batch_size = self.params.batch_size
        window_size = self.params.window_size
        vocab_size = self.params.vocab_size
        embedding_size = self.params.embedding_size
        hidden_size = self.params.hidden_size
        smoothing_window_size = self.params.smoothing_window_size
        variable_init = self.params.variable_init
        
        self.E = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='E')
        self.U_w = tf.Variable(tf.truncated_normal((window_size*embedding_size, hidden_size),
                                                   stddev=1.0/math.sqrt(hidden_size)), name='U_w')
        self.U_b = tf.Variable(tf.zeros([hidden_size]), name='U_b')
        self.V_w = tf.Variable(tf.truncated_normal((hidden_size, vocab_size),
                                                   stddev=1.0/math.sqrt(vocab_size)), name='V_w')
        self.V_b = tf.Variable(tf.zeros([vocab_size]), name='V_b')
        self.W_w = tf.Variable(tf.truncated_normal((hidden_size, vocab_size),
                                                   stddev=1.0/math.sqrt(vocab_size)), name='W_w')
        self.W_b = tf.Variable(tf.zeros([vocab_size]), name='W_b')
        self.G = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='G')
        self.F = tf.Variable(tf.random_uniform([vocab_size, hidden_size], -1.0, 1.0), name='F')
        self.P = tf.Variable(tf.truncated_normal((window_size*embedding_size, hidden_size),
                                                 stddev=1.0/math.sqrt(hidden_size)), name='P')

        restore_vals = {'E': self.E,
                        'U_w': self.U_w,
                        'U_b': self.U_b,
                        'V_w': self.V_w,
                        'V_b': self.V_b,
                        'W_w': self.W_w,
                        'W_b': self.W_b,
                        'G': self.G,
                        'F': self.F,
                        'P': self.P}
        
        saver = tf.train.Saver(restore_vals)
        saver.restore(sess, model_path)

        self.y_c = tf.placeholder(tf.int32, shape=[batch_size, window_size]) 
        self.tilde_y_c = tf.reshape(tf.nn.embedding_lookup(self.E, self.y_c), shape=[batch_size, window_size*embedding_size])

        self.h = tf.nn.tanh(tf.matmul(self.tilde_y_c, self.U_w) + self.U_b)

        self.prob_from_h = tf.matmul(self.h, self.V_w) + self.V_b


        self.x = tf.placeholder(tf.int32, shape=[batch_size, None])


        self.tilde_y_c_dash = tf.reshape(tf.nn.embedding_lookup(self.G, self.y_c), shape=[batch_size, window_size*embedding_size])

        self.tilde_x = tf.cast(tf.nn.embedding_lookup(self.F, self.x), tf.float32)

        self.p_ = tf.matmul(self.tilde_y_c_dash, self.P)

        self.tilde_x = tf.transpose(self.tilde_x, [1, 0, 2])

        initializer = [tf.reshape(tf.slice(self.tilde_x, [0, 0, 0], [1, -1, -1]), [batch_size, hidden_size])] * (smoothing_window_size+1)
        x_bar_forward = tf.scan(lambda a, x: a[1: -1] + [x] + [sum(a[: -1] + [x])], elems=self.tilde_x, initializer=initializer)[-1]
        
        reverse_tilde_x = tf.reverse(self.tilde_x, [True, False, False])
        
        initializer = [tf.reshape(tf.slice(reverse_tilde_x, [0, 0, 0], [1, -1, -1]), [batch_size, hidden_size])] * (smoothing_window_size+1)
        x_bar_backward = tf.reverse(tf.scan(lambda a, x: a[1: -1] + [x] + [sum(a[: -1] + [x])], elems=reverse_tilde_x, initializer=initializer)[-1], [True, False, False])

        self.x_bar = tf.transpose((x_bar_forward + x_bar_backward - self.tilde_x), [1, 0, 2]) / (2*smoothing_window_size+1)
        
        self.tilde_x = tf.transpose(self.tilde_x, [1, 0, 2])

        self.enc = tf.reshape(tf.scan(lambda a, x: tf.matmul(tf.nn.softmax(tf.matmul(tf.expand_dims(x[0], 0), x[1])), x[2]),
                                      elems=(self.p_, tf.transpose(self.tilde_x, [0, 2, 1]), self.x_bar),
                                      initializer=tf.zeros([1, hidden_size])), [batch_size, hidden_size])
        
        self.prob_from_enc = tf.matmul(self.enc, self.W_w) + self.W_b

        self.prob = tf.nn.softmax(self.prob_from_h + self.prob_from_enc)
        self.pred_prob = tf.reduce_max(self.prob, 1)[0]
        self.pred = tf.argmax(self.prob, 1)[0]

        # ### for training ###
        self.t = tf.cast(tf.placeholder(tf.int32, shape=[batch_size, vocab_size]), tf.float32)
        self.cross_entropy = -tf.reduce_sum(self.t * tf.log(self.prob))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.prob, 1), tf.argmax(self.t, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def rebuild_forward_graph(self, sess, model_path):
        self.params.batch_size = 1
        self.rebuild_graph(sess, model_path)
        
    def decode(self, session, x, y_c):
        feed_dict = {self.x: x, self.y_c: y_c}
        return session.run([self.pred, self.pred_prob, self.prob], feed_dict=feed_dict)
        
        
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
    
   
