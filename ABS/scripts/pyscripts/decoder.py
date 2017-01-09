# -*- coding: utf-8 -*-

import os
import sys
import argparse
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from models import BOWmodel, ABSmodel
import reuters_dataset

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--model_path', type=str)
args = parser.parse_args()

sess = tf.Session()
saver.restore(sess, args.model_path)

