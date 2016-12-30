import tensorflow as tf
import numpy as np

class sentenceEncorder(object):
    def __init__(self, sequence_length, num_classes, vocab_size,　embedding_size, filter_sizes, num_filters):
        # sequence_lengthは固定にしましょう（実証上難しそう）
        # embedding_sizeはword2vec使うので300固定ですがまあこのままにしておきます．
        # このencoderを別に学習させるか，全体で学習させるかまだ決まってないですよね
        # 多分いらないと思いますが…一旦作っておきます．
        self.train_X = tf.placeholder(tf.int32, [None, sequence_length], name = "train_X")
        self.ftune_y = tf.placeholder(tf.float32, [None, num_classes], name = "ftune_y")
        l2_loss = tf.constant(0.1)

        # Embedding
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.train_X)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Convolution and Max-pooling
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,W,
                    strides=[1, 1, 1, 1],padding="VALID",name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(
                    h, ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],padding='VALID',name="pool")
                pooled_outputs.append(pooled)

        # Max pooling の 結果を Concanate , Featureを作る
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Dropout
        self.dropout_rate = tf.placeholder(tf.float32, name = "dropout_rate")
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_rate)

        # Fully Connected layer with Xavier (これで良いかはまだわからん)
        with tf.name_scope("output"):
            W = tf.get_variable("W",shape=[num_filters_total, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        # with tf.name_scope("loss"):
        #    losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.ftune_y)
        #    self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        # with tf.name_scope("accuracy"):
        #    correct_predictions = tf.equal(self.predictions, tf.argmax(self.ftune_y, 1))
        #    self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
