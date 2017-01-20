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
        # C: window_size
        # V: vocab_size
        # D: embedding_size

        self.y_c = tf.placeholder(tf.float32, shape=[batch_size, window_size*embedding_size])
        self.E_w = tf.Variable(tf.truncated_normal((window_size*embedding_size ,window_size*embedding_size),
                                                   stddev=variable_init/math.sqrt(window_size*embedding_size)), name='E_w')
        self.E_b = tf.Variable(tf.zeros([window_size*embedding_size]), name='E_b')
        self.tilde_y_c = tf.nn.tanh(tf.matmul(self.y_c, self.E_w) + self.E_b)
        
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

        self.x = tf.placeholder(tf.float32, shape=[batch_size, None, embedding_size])
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

        self.F_w = tf.Variable(tf.truncated_normal((embedding_size, hidden_size),
                                                   stddev=variable_init/math.sqrt(hidden_size)), name='F_w')
        self.F_b = tf.Variable(tf.zeros([hidden_size]), name='F_b')
        
        initializer = tf.zeros([batch_size, hidden_size])
        self.tilde_x = tf.scan(lambda a, x: tf.nn.tanh(tf.matmul(x, self.F_w)+self.F_b),
                               elems=tf.transpose(self.x, [1, 0, 2]), # (?, batch_size, embedding_size)
                               initializer=initializer) # (?, batch_size, hidden_size)

        self.G_w = tf.Variable(tf.truncated_normal((window_size*embedding_size ,window_size*embedding_size),
                                                   stddev=variable_init/math.sqrt(window_size*embedding_size)), name='G_w')
        self.G_b = tf.Variable(tf.zeros([window_size*embedding_size]), name='G_b')
        self.tilde_y_c_dash = tf.nn.tanh(tf.matmul(self.y_c, self.G_w) + self.G_b)
        self.P = tf.Variable(tf.truncated_normal((window_size*embedding_size, hidden_size),
                                                 stddev=variable_init/math.sqrt(hidden_size)), name='P')
        self.p_ = tf.matmul(self.tilde_y_c_dash, self.P)

        initializer = [tf.reshape(tf.slice(self.tilde_x, [0, 0, 0], [1, -1, -1]), [batch_size, hidden_size])] * (smoothing_window_size+1) # 前window_size分は元のやつ、残り一つ分が欲しいやつ
        x_bar_forward = tf.scan(lambda a, x: a[1: -1] + [x] + [sum(a[: -1] + [x])], elems=self.tilde_x, initializer=initializer)[-1]
        
        reverse_tilde_x = tf.reverse(self.tilde_x, [True, False, False]) # reverseの扱い注意
        
        initializer = [tf.reshape(tf.slice(reverse_tilde_x, [0, 0, 0], [1, -1, -1]), [batch_size, hidden_size])] * (smoothing_window_size+1)
        x_bar_backward = tf.reverse(tf.scan(lambda a, x: a[1: -1] + [x] + [sum(a[: -1] + [x])], elems=reverse_tilde_x, initializer=initializer)[-1], [True, False, False])

        self.x_bar = tf.transpose((x_bar_forward + x_bar_backward - self.tilde_x), [1, 0, 2]) / (2*smoothing_window_size+1)
        
        self.tilde_x = tf.transpose(self.tilde_x, [1, 0, 2]) # x_barを計算した後

        print(self.tilde_x.get_shape()) # (64, ?, 400)
        print(self.x_bar.get_shape()) # (64, ?, 400)
        print(self.p_.get_shape()) # (64, 400)
        
        self.enc = tf.reshape(tf.scan(lambda a, x: tf.matmul(tf.nn.softmax(tf.matmul(tf.expand_dims(x[0], 0), x[1])), x[2]), #(1, ?) (?, hidden_size)
                                      elems=(self.p_, tf.transpose(self.tilde_x, [0, 2, 1]), self.x_bar),
                                      initializer=tf.zeros([1, hidden_size])), [batch_size, hidden_size])

                
    def train(self, session, x, y_c, t):
        feed_dict = {self.x: x, self.y_c: y_c, self.t: t}
        self.train_step.run(session=session, feed_dict=feed_dict)

    
    def rebuild_graph(self, sess, model_path, forward=True, var_list=None):

        batch_size = self.params.batch_size
        window_size = self.params.window_size
        vocab_size = self.params.vocab_size
        embedding_size = self.params.embedding_size
        hidden_size = self.params.hidden_size
        smoothing_window_size = self.params.smoothing_window_size 
        variable_init = self.params.variable_init
        
        self.E_w = tf.Variable(tf.truncated_normal((window_size*embedding_size ,window_size*embedding_size),
                                                   stddev=variable_init/math.sqrt(window_size*embedding_size)), name='E_w')
        self.E_b = tf.Variable(tf.zeros([window_size*embedding_size]), name='E_b')
        self.F_w = tf.Variable(tf.truncated_normal((embedding_size, hidden_size),
                                                   stddev=variable_init/math.sqrt(hidden_size)), name='F_w')
        self.F_b = tf.Variable(tf.zeros([hidden_size]), name='F_b')
        self.G_w = tf.Variable(tf.truncated_normal((window_size*embedding_size ,window_size*embedding_size),
                                                   stddev=variable_init/math.sqrt(window_size*embedding_size)), name='G_w')
        self.G_b = tf.Variable(tf.zeros([window_size*embedding_size]), name='G_b')
        self.U_w = tf.Variable(tf.truncated_normal((window_size*embedding_size, hidden_size),
                                                   stddev=1.0/math.sqrt(hidden_size)), name='U_w')
        self.U_b = tf.Variable(tf.zeros([hidden_size]), name='U_b')
        self.V_w = tf.Variable(tf.truncated_normal((hidden_size, vocab_size),
                                                   stddev=1.0/math.sqrt(vocab_size)), name='V_w')
        self.V_b = tf.Variable(tf.zeros([vocab_size]), name='V_b')
        self.W_w = tf.Variable(tf.truncated_normal((hidden_size, vocab_size),
                                                   stddev=1.0/math.sqrt(vocab_size)), name='W_w')
        self.W_b = tf.Variable(tf.zeros([vocab_size]), name='W_b')
        self.P = tf.Variable(tf.truncated_normal((window_size*embedding_size, hidden_size),
                                                 stddev=1.0/math.sqrt(hidden_size)), name='P')

        restore_vals = {'E_w': self.E_w,
                        'E_b': self.E_b,
                        'F_w': self.F_w,
                        'F_b': self.F_b,
                        'G_w': self.G_w,
                        'G_b': self.G_b,
                        'U_w': self.U_w,
                        'U_b': self.U_b,
                        'V_w': self.V_w,
                        'V_b': self.V_b,
                        'W_w': self.W_w,
                        'W_b': self.W_b,
                        'P': self.P}
        
        saver = tf.train.Saver(restore_vals)
        saver.restore(sess, model_path)

        self.y_c = tf.placeholder(tf.float32, shape=[batch_size, window_size*embedding_size])
        self.tilde_y_c = tf.nn.tanh(tf.matmul(self.y_c, self.E_w) + self.E_b)
     
        self.h = tf.nn.tanh(tf.matmul(self.tilde_y_c, self.U_w) + self.U_b)
        
        self.prob_from_h = tf.matmul(self.h, self.V_w) + self.V_b
        
        self.x = tf.placeholder(tf.float32, shape=[batch_size, None, embedding_size])
        
        initializer = tf.zeros([batch_size, hidden_size])
        self.tilde_x = tf.scan(lambda a, x: tf.nn.tanh(tf.matmul(x, self.F_w)+self.F_b),
                               elems=tf.transpose(self.x, [1, 0, 2]),
                               initializer=initializer)

        self.tilde_y_c_dash = tf.nn.tanh(tf.matmul(self.y_c, self.G_w) + self.G_b)

        self.p_ = tf.matmul(self.tilde_y_c_dash, self.P)

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

        if forward:
            self.x_weight = tf.placeholder(tf.float32, shape=[vocab_size])
            self.prob = tf.nn.softmax(self.prob_from_h + self.prob_from_enc + self.x_weight)
        else:
            self.prob = tf.nn.softmax(self.prob_from_h + self.prob_from_enc)
            
    def rebuild_forward_graph(self, sess, model_path):
        self.params.batch_size = 1
        self.rebuild_graph(sess, model_path)
        
    def decode(self, session, x, y_c, x_weight):
        feed_dict = {self.x: x, self.y_c: y_c, self.x_weight: x_weight}
        return session.run(self.prob, feed_dict=feed_dict)

class ABS_plus_model(ABSmodel):

    def build_train_graph_for_alpha(self, sess, model_path):
        batch_size = self.params.batch_size
        vocab_size = self.params.vocab_size
        self.alpha = tf.Variable(tf.ones([5]), dtype=tf.float32, name='alpha')
        
        self.features = tf.placeholder(tf.float32, shape=[batch_size, vocab_size, 4]) 

        self.rebuild_graph(sess, model_path, forward=False, var_list=[self.alpha])

        self.f = tf.concat(2, [tf.expand_dims(self.prob, 2), self.features]) #(batch_size, vocab_size, 5)
        
        initializer = tf.zeros([1, vocab_size])
        self.prob_plus = tf.nn.softmax(tf.reshape(tf.scan(lambda a, x: tf.matmul(tf.expand_dims(self.alpha, 0), x),
                                                          elems=tf.transpose(self.f, [0, 2, 1]),
                                                          initializer=initializer), [batch_size, -1]))
    
        ### for training ###
        self.t = tf.cast(tf.placeholder(tf.int32, shape=[batch_size, vocab_size]), tf.float32)
        self.cross_entropy = -tf.reduce_sum(self.t * tf.log(self.prob_plus)) 
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy, var_list=[self.alpha])
        self.correct_prediction = tf.equal(tf.argmax(self.prob_plus, 1), tf.argmax(self.t, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def train(self, session, x, y_c, features, t):
        feed_dict = {self.x: x, self.y_c: y_c, self.features: features, self.t: t}
        self.train_step.run(session=session, feed_dict=feed_dict)
        
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
    
   
