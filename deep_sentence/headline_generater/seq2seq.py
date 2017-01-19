from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging
import argparse

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from sklearn.model_selection import train_test_split

from headline_generator import HeadlineGenerator
from util import make_dataset

try:
  tf.app.flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32.")
  tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
  tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
  tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
  tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
  tf.app.flags.DEFINE_integer("size", 200, "Size of each model layer.")
  tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
  tf.app.flags.DEFINE_integer("vocab_size", 1005367, "Vocabulary size.")
  tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
  tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
  tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
  tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")
  tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
except argparse.ArgumentError:
  pass

FLAGS = tf.app.flags.FLAGS


def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  model = HeadlineGenerator(
        vocab_size=FLAGS.vocab_size,
        size=FLAGS.size,
        max_gradient_norm=FLAGS.max_gradient_norm,
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate,
        learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
        forward_only=forward_only,
        dtype=dtype,
        num_layers=FLAGS.num_layers,
  )
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model


def train(source_ids, target_ids, dic, vec):
  with tf.Session() as sess:
    # Read embeddings.
    # embeddings = tf.nn.embedding_lookup()

    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False)

    # Read data and compute their size.
    from_train, to_train, from_dev, to_dev = train_test_split(source_ids, target_ids, test_size=0.1)
    train_set = [from_train, to_train]
    dev_set = [from_dev, to_dev]
    train_size = len(train_set)

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set)

      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        # Run evals on development set and print their perplexity.
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set)
        _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, True)
        eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
        print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()


def main(_):
  if FLAGS.decode:
    pass
    # decode()
  else:
    source_ids, target_ids, dic, vec = make_dataset()
    train(source_ids, target_ids, dic, vec)


if __name__ == "__main__":
  tf.app.run()
