# -*- coding: utf-8 -*-

import argparse
import pickle
from getpass import getpass
import psycopg2
import numpy as np
import pandas as pd
import MeCab
from gensim.models import word2vec

from fetch_rawdata import fetch

def tokenize_POSfilter(sentence):
    tagger = MeCab.Tagger('-Ochasen')
    tagger.parse('')
    node = tagger.parseToNode(sentence)
    filtered = []
    while node:
        if str(node.feature.split(',')[0]) in WORD_CLASS and not node.surface in STOP:
            filtered.append(node.surface)
        node = node.next
    return filtered

def tokenize(sentence):
    mt    = MeCab.Tagger('-Owakati') 
    parse = mt.parse(sentence)
    return parse.split()

if __name__ == '__main__':

    pd.set_option('display.width', 1000)
    
    parser = argparse.ArgumentParser(description='fetch rawdata from database')
    parser.add_argument('--password', type=str, default=None,
                        help='remove from git management')
    parser.add_argument('--n_rows', type=int, default=None,
                        help='number of row to select')
    parser.add_argument('--size', type=int, default=50, help='word2vec size')
    parser.add_argument('--window', type=int, default=5, help='word2vec window')
    parser.add_argument('--min_count', type=int, default=5, help='word2vec minimum count')
    parser.add_argument('model_path', type=str, help='output filename')
    
    args = parser.parse_args()

    if args.password:
        pw = args.password
    else:
        pw = getpass('database password: ')

    print('fetching articles...')
    F = fetch(pw)
    F.get_articles(args.n_rows)
    F.get_(args.n_rows)
    corpus = list(F.articles['content'])
    #corpus.extend(list(F.sources['content']))
    
    print('training word2vec model...')
    token = list(map(tokenize, corpus))
    model = word2vec.Word2Vec(token,
                              size=args.size,
                              window=args.window,
                              min_count=args.min_count)
    
    with open(args.model_path, 'wb') as f_model:
        pickle.dump(model, f_model)
    
    
