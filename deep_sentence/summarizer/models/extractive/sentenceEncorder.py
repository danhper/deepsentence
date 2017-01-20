#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from base import Model
from utils import *

class sentenceEncorder(Model):
    # based on https://arxiv.org/pdf/1508.06615v4.pdf (2016, KIM yoon)
    def __init__(self, input_, embed_dim=150,
                 feature_maps=[10, 20, 30, 60, 60, 60, 60],
                 #ここ論文の表現曖昧ですが… 修正するかもしれません．SentenceEmbeddingが300になるようにしておきます．
                 kernels=[1,2,3,4,5,6,7]):
        self.embed_dim = embed_dim
        self.feature_maps = feature_maps
        self.kernels = kernels

        input_ = tf.expand_dims(input_, -1)
        layers = []

        for idx, kernel_dim in enumerate(kernels):
            reduced_length = input_.get_shape()[1] - kernel_dim + 1
            with tf.variable_scope('conv2d'):
                w_ = tf.get_variable('w',[kernel_dim, self.embed_dim, input_.get_shape()[-1], feature_maps[idx]],
                                     initializer=tf.truncated_normal_initializer(stddev=0.01))
                conv = tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID')
            # time-delayed-maxpooling
            pool = tf.nn.max_pool(tf.tanh(conv), [1, reduced_length, 1, 1], [1, 1, 1, 1], 'VALID')
            layers.append(tf.squeeze(pool))
            if len(kernels) > 1:
                self.output = tf.concat(1, layers)
            else:
                self.output = layers[0]
