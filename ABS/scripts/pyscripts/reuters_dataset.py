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

    # data = data.iloc[: 500] #################################################################################################3
    
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
    sentence = first_sentence(series['content'])
    if sentence:
        token = tokenize(sentence)
    else:
        return None
    return token

def tokenize_title(series, window_size):
    token = tokenize(series['title'])
    return ['<S>']*window_size + token + ['EOS']

def labeling(token, dictionary):
    return list(map(lambda x: dictionary.token2id[x], token))
      
def split_yc_t(arr, window_size):
    x_labels = list(map(int, arr[0].replace('[', '').replace(']', '').split(',')))
    yc_and_t_labels = list(map(int, arr[1].replace('[', '').replace(']', '').split(',')))
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

    return result.values

def filtering(dataset):
    dataset = dataset.drop(np.where(dataset['x_length']<=dataset['t_length'])[0], axis=0)
    return dataset
    
def arrange_train(dataset, params, dictionary):
    print('arranging train...')
    window_size = params.window_size
    dataset = dataset.assign(x_tokens=lambda dataset: dataset.apply(tokenize_first_sentence, axis=1),
                             t_tokens=lambda dataset: dataset.apply(functools.partial(tokenize_title,
                                                                                      window_size=window_size), axis=1),
    ).drop(['title', 'content'], axis=1
    ).dropna(axis=0
    )[['x_tokens', 't_tokens']]

    dataset.reset_index(inplace=True, drop=True)
    
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

    dataset = dataset.sort_values('x_length')
    dataset.reset_index(inplace=True, drop=True)

    return dataset, dictionary

def arrange_test(dataset, params, dictionary):
    print('arranging test...')
    window_size = params.window_size
    dataset = dataset.assign(x_tokens=lambda dataset: dataset.apply(tokenize_first_sentence, axis=1),
                             t_tokens=lambda dataset: dataset.apply(functools.partial(tokenize_title,
                                                                                      window_size=window_size), axis=1),
    ).drop(['title', 'content'], axis=1
    ).dropna(axis=0
    )[['x_tokens', 't_tokens']]

    dataset.reset_index(inplace=True, drop=True)

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
    dataset = dataset.drop(['x_length', 't_length'], axis=1)

    return dataset, dictionary

def pad_x(x_labels, x_max_length, symbol_ids):
    return x_labels + [symbol_ids['EOS']]*(x_max_length-len(x_labels))

def make_p(x_length, x_max_length):
    return [1.0/x_length]*x_length + [0.0]*(x_max_length-x_length)                                     

def make_batch(dataset, symbol_ids, params):
    batch_size = params.batch_size
    window_size =params.window_size
    
    pool = Pool(8)
    results = pool.map(functools.partial(split_yc_t,
                                         window_size=window_size), dataset[['x_labels', 'yc_and_t_labels']].values)
    pool.close()
    
    for j, result in enumerate(results):
        if j == 0:
            results = result
        else:
            results = np.r_[results, result] 
            
    dataset = pd.DataFrame(results, columns=['x_labels', 'yc_labels', 't_label'])
    dataset = dataset.assign(x_length=lambda dataset: dataset['x_labels'].apply(lambda x: len(x)))
    
    n_batch = int((dataset.shape[0]-1)/batch_size+1)-1 ############################################## -1
    
    for i in range(n_batch):
        batch_df = dataset.iloc[i*batch_size: (i+1)*batch_size]
        x_max_length = batch_df['x_length'].max()
        batch_df = batch_df.assign(x_labels_padded=lambda batch_df: batch_df['x_labels'].apply(functools.partial(pad_x,
                                                                                                                 x_max_length=x_max_length,
                                                                                                                 symbol_ids=symbol_ids)),
        ).drop(['x_labels', 'x_length'], axis=1
        ).rename(columns={'x_labels_padded': 'x_labels'}
        )[['x_labels', 'yc_labels', 't_label']]
        
        # print(batch_df.assign(length=lambda batch_df: batch_df['x_labels'].apply(lambda x: len(x))))
        
        yield i, n_batch, batch_df
   

def save_dataset(path, dataset):
    print('saving dataset...')
    dataset.to_csv(path, index_label=False)

def load_dataset(path, skiprows, nrows):
    col_names = ['x_labels', 'yc_and_t_labels', 't_length', 'x_length']
    with open(path, 'r') as f_load:
        dataset = pd.read_csv(f_load, skiprows=skiprows, nrows=nrows, names=col_names)
    return dataset

def save_dictionary(path):
    print('saving dictionary...')
    with open(path, 'wb') as f_dict:
        pickle.dump(dictionary, f_dict)

def load_dictionary(path):
    print('loading dictionary...')
    with open(path, 'rb') as f_dict:
        return pickle.load(f_dict)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_paths_template', type=str, default=None)
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()
    
    pd.set_option('display.width', 10000)

    data_paths = glob.glob(args.data_paths_template.replace('\\', ''))
    
    data = load_rawdata(data_paths)
    train, test =  split_data(data)
    dictionary = None
    train_dataset, dictionary = arrange_train(train, config.params, dictionary)
    test_dataset, dictionary = arrange_test(test, config.params, dictionary)
    save_dataset(args.save_dir+'/train.csv', train_dataset)
    save_dataset(args.save_dir+'/test.csv', test_dataset)
    save_dictionary(args.save_dir+'/dictionary.pkl')

