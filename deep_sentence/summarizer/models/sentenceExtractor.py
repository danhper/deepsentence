#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import tensorflow as tf

from base import Model
from utils import progress
from sentenceEncorder import sentenceEncorder

class sentenceExtractor(Model):
    def __init__(self, sess, word_vocab_size, decoder_inputs, batch_size=20,
                 rnn_size=650, layer_depth=1, word_embed_dim=150,
                 feature_maps=[10, 20, 30, 60, 60, 60, 60], kernels=[1,2,3,4,5,6,7],
                 max_grad_norm=5, dropout_prob=0.5):

        self.dropout_prob = dropout_prob
        self.data_dir = "data"
        self.dataset_name = "mitei"
        self.checkpoint_dir = "checkpoint"
        self.batch_size = batch_size
        self.seq_length = 60 # length of sequence
        self.decoder_inputs = decoder_inputs

        # RNN
        self.sess = sess
        self.rnn_size = rnn_size
        self.layer_depth = layer_depth
        self.num_decoder_symbols = 2

        # CNN
        self.word_vocab_size = word_vocab_size
        self.word_embed_dim = word_embed_dim
        self.feature_maps = feature_maps
        self.kernels = kernels

        # data_loading
        # self.loader = BatchLoader()

    def prepare_model(self):
        with tf.variable_scope("sentenceExtractor"):
            self.word_inputs = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])
            self.decoder_inputs = []

            with tf.variable_scope("embedding") as scope:
                embedding_W = tf.get_variable("word_embedding", [self.word_vocab_size, self.word_embed_dim])
                self.embedded_word = tf.nn.embedding_lookup(W, self.word_inputs)
                # word2vec?

            self.cnn_outputs = []

            with tf.variable_scope("sentenceEncorder") as scope:
                word_indices = tf.split(1, self.seq_length, tf.expand_dims(self.word_inputs, -1))
                for idx in xrange(self.seq_length):
                    word_index = tf.reshape(word_indices[idx], [-1, 1])
                    if idx != 0:
                        scope.reuse_variables()

                    word_cnn = sentenceEncorder(self.embedded_word, self.word_embed_dim, self.feature_maps, self.kernels)
                    cnn_output = word_cnn.output
                    self.cnn_outputs.append(cnn_output)

                #######################################################################
                #bn = batch_norm()
                #norm_output = bn(tf.expand_dims(tf.expand_dims(cnn_output, 1), 1))
                #cnn_output = tf.squeeze(norm_output)
                #self.cnn_outputs.append(cnn_output)
                #######################################################################

            # ... #
            with tf.variable_scope("LSTM") as scope:
                self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)
                self.stacked_cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * self.layer_depth)
                outputs, _ = tf.nn.seq2seq.embedding_attention_seq2seq(self.cnn_outputs,
                                                                       self.decoder_inputs,
                                                                       self.cell,
                                                                       vocabulary_size,
                                                                       self.num_decoder_symbols,
                                                                       300, # size of cnn_output
                                                                       feed_previous=feed_previous)

                self.lstm_outputs = []
                self.true_outputs = tf.placeholder(tf.int64,[self.batch_size, self.seq_length])
                true_outputs = tf.split(1, self.seq_length, self.true_outputs)

                loss = 0

            for idx, (top_h, true_output) in enumerate(zip(outputs, true_outputs)):
                top_h = tf.nn.dropout(top_h, self.dropout_prob)
                self.lstm_outputs.append(top_h)
                loss += tf.nn.sparse_softmax_cross_entropy_with_logits(self.lstm_outputs[idx], tf.squeeze(true_output))

            self.loss = tf.reduce_mean(loss) / self.seq_length
            tf.scalar_summary("loss", self.loss)

def train(self, epoch):
    cost = 0
    target = np.zeros([self.batch_size, self.seq_length])
    N = self.loader.sizes[0]
    for idx in xrange(N):
        target.fill(0)
        x, y, x_char = self.loader.next_batch(0)
        for b in xrange(self.batch_size):
            for t, w in enumerate(y[b]):
                target[b][t] = w

        feed_dict = {
          self.word_inputs: x,
          self.char_inputs: x_char,
          self.true_outputs: target,
        }

        _, loss, step, summary_str = self.sess.run(
            [self.optim, self.loss, self.global_step, self.merged_summary], feed_dict=feed_dict)

        self.writer.add_summary(summary_str, step)

        if idx % 50 == 0:
            progress(idx/N, "epoch: [%2d] [%4d/%4d] loss: %2.6f" % (epoch, idx, N, loss))

        cost += loss

    return cost / N
