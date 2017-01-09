# -*- coding: utf-8 -*-

import os
import sys
import argparse
import re
import glob
import pickle
import functools
from multiprocessing import Pool

import MeCab
import gensim
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

import config

def load_rawdata(data_paths):
    print('loading... (file num = %s)'%len(data_paths))
    first_head_data = True
    for path in data_paths:
        try:
            temp = pd.read_csv(path, usecols=[5, 6]).dropna(axis=0)
        except pd.io.common.EmptyDataError:
            continue
        if first_head_data:
            data = temp
            first_head_data = False
        else:
            data = pd.concat((data, temp), axis=0)

    data.reset_index(inplace=True, drop=True)
    print('data shape', data.shape)
    # print(data)

    data = data.iloc[: 10] #################################################################################################3
    
    return data

def split_data(data):
    train, test = train_test_split(data, test_size=0.20, random_state=0)
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)
    return train, test
    

def first_sentence(sentences):
    try:
        first = re.match('(.*?)。', sentences).group(0)
    except AttributeError:
        return None
    m = re.search('］ - (.*?)。', first)
    if not m:
        m = re.search('］ (.*?)。', first)
    if not m:
        m = re.search('】(.*?)。', first)
    if not m:
        m = re.search('\ (.*?)。', first)
    if not m:
        m = re.search('^(.*?)。', first)
    if not m:
        m = re.search('(.*?)。', first)
    return m.group(1)
    
def tokenize(sentence):
    mt    = MeCab.Tagger('-Owakati') 
    parse = mt.parse(sentence)
    return parse.split()

def tokenize_first_sentence(series):
    # print('========================================')
    # print(series['content'])
    sentence = first_sentence(series['content'])
    # print('-------------------------------')
    # print(sentence)
    if sentence:
        token = tokenize(sentence)
    else:
        # import time
        # time.sleep(10)
        return None
    # print('-------------------------------')
    # print(token)
    
    return token

def tokenize_title(series, window_size):
    token = tokenize(series['title'])
    return ['<S>']*window_size + token + ['EOS']

def labeling(token, dictionary):
    return list(map(lambda x: dictionary.token2id[x], token))
      
def split_yc_t(arr, window_size, dictionary):    
    x_labels = arr[0]
    yc_and_t_labels = arr[1]
    for i in range(len(yc_and_t_labels[: -window_size])):
        yc_labels = yc_and_t_labels[i: i+window_size]
        t_label = yc_and_t_labels[i+window_size]
        temp = pd.DataFrame([[x_labels, yc_labels, t_label]])
        if i == 0:
            result = temp
        else:
            result = pd.concat((result, temp), axis=0)
    result.reset_index(inplace=True, drop=True)
    result.columns = ['x_labels', 'yc_labels', 't_label']

    # print(result)
    
    return result.values

def filtering(dataset):
    dataset = dataset.drop(np.where(dataset['x_length']<=dataset['t_length'])[0], axis=0)
    return dataset
    
def arrange(dataset, params, dictionary):
    print('arranging...')
    window_size = params.window_size
    dataset = dataset.assign(x_tokens=lambda dataset: dataset.apply(tokenize_first_sentence, axis=1),
                             t_tokens=lambda dataset: dataset.apply(functools.partial(tokenize_title,
                                                                                      window_size=window_size), axis=1),
    ).drop(['title', 'content'], axis=1
    ).dropna(axis=0
    )[['x_tokens', 't_tokens']]

    dataset.reset_index(inplace=True, drop=True)
    
    # print(dataset)

    if dictionary:
        dictionary.add_documents(dataset.values.flatten())
    else:
        dictionary = gensim.corpora.Dictionary(dataset.values.flatten())
    print('vocabulary size:', len(list(dictionary.iterkeys())))
    
    dataset = dataset.applymap(functools.partial(labeling, dictionary=dictionary)
    ).assign(x_length=lambda dataset: dataset['x_tokens'].apply(lambda x: len(x)),
             t_length=lambda dataset: dataset['t_tokens'].apply(lambda x: len(x)-window_size-1)
    ).rename(columns={'x_tokens': 'x_labels', 't_tokens': 'yc_and_t_labels'})

    dataset = filtering(dataset)

    # print(dataset)

    dataset = dataset.sort_values('x_length')
    dataset.reset_index(inplace=True, drop=True)
    
    pool = Pool()
    results = pool.map(functools.partial(split_yc_t,
                                         window_size=window_size,
                                         dictionary=dictionary), dataset.values)
    for i, result in enumerate(results):
        if i == 0:
            results = result
        else:
            results = np.r_[results, result]    
            
    dataset = pd.DataFrame(results, columns=['x_labels', 'yc_labels', 't_label']
    ).assign(x_length=lambda dataset: dataset['x_labels'].apply(lambda x: len(x)))

    # print(dataset)
    
    return dataset, dictionary

def pad_x(x_labels, x_max_length, dictionary):
    # if len(x_labels) != x_max_length:
        # print(np.r_[x_labels, np.array([dictionary.token2id['EOS']]*(x_max_length-len(x_labels)))])
    return x_labels + [dictionary.token2id['EOS']]*(x_max_length-len(x_labels))
    #return np.r_[x_labels, np.array([dictionary.token2id['EOS']]*(x_max_length-len(x_labels)))]
   

def make_p(x_length, x_max_length):
    return [1.0/x_length]*x_length + [0.0]*(x_max_length-x_length)
                                           

def make_batch(dataset, dictionary, params):
    print('making batch...')
    batch_size = params.batch_size
    n_batch = int((dataset.shape[0]-1)/batch_size+1)

    list_batch = []

    #dataset = dataset.applymap(lambda x: np.array(x).astype(np.int32))
    
    for i in range(n_batch):
        batch_df = dataset[i*batch_size: (i+1)*batch_size]
        x_max_length = batch_df['x_length'].max()
        batch_df = batch_df.assign(x_labels_padded=lambda batch_df: batch_df['x_labels'].apply(functools.partial(pad_x,
                                                                                                      x_max_length=x_max_length,
                                                                                                      dictionary=dictionary))
        ).drop('x_labels', axis=1
        ).rename(columns={'x_labels_padded': 'x_labels'})
        
        batch_df = batch_df.assign(p=lambda batch_df: batch_df['x_length'].apply(functools.partial(make_p,
                                                                                                   x_max_length=x_max_length))
        ).drop('x_length', axis=1)
        
        #print(batch_df.assign(length=lambda batch_df: batch_df['x_labels'].apply(lambda x: len(x))))
        # print(batch_df)
        
        #yield batch_df.values
        list_batch.append(batch_df)
    return list_batch

def save_dataset(path, dataset):
    print('saving dataset...')
    dataset.to_pickle(path)

def load_dataset(path):
    print('loding dataset...')
    dataset = pd.read_pickle(path)
    return dataset
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_paths_template', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()
    
    pd.set_option('display.width', 10000)

    data_paths = glob.glob(args.data_paths_template.replace('\\', ''))
    
    data = load_rawdata(data_paths)
    train, test =  split_data(data)
    dictionary = None
    train_dataset, dictionary = arrange(train, config.params, dictionary)
    test_dataset, dictionary = arrange(test, config.params, dictionary)
    #make_batch(dataset, dictionary, config.params)
    save_dataset(args.save_dir+'/train.pkl', train_dataset)
    save_dataset(args.save_dir+'/test.pkl', test_dataset)
    with open(args.save_dir+'/dictionary.pkl', 'wb') as f_dict:
        pickle.dump(dictionary, f_dict)
    
    load_dataset(args.save_dir+'/train.pkl')
    a
