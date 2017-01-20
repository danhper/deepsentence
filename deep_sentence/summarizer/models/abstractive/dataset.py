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

import reuters
import from_db

import config

def split_train_test(data):
    train, test = train_test_split(data, test_size=0.10, random_state=0)
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)
    return train, test
    
def tokenize(sentence):
    mt    = MeCab.Tagger('-Owakati') 
    parse = mt.parse(sentence)
    return parse.split()

def tokenize_title(title, window_size):
    token = tokenize(title)
    return ['<S>']*window_size + token + ['<EOS>']

def labeling(token, dictionary):
    def func(x):
        try:
            return dictionary[x]
        except KeyError:
            return dictionary['<UNK>']
    return list(map(func, token))
      
def filtering(dataset):
    dataset = dataset.drop(np.where(dataset['x_length']<=dataset['t_length'])[0], axis=0)
    return dataset
    
def arrange(dataset, params, dictionary):
    print('arranging dataset...')
    window_size = params.window_size
    dataset = dataset.assign(x_tokens=lambda dataset: dataset['first'].apply(tokenize),
                             t_tokens=lambda dataset: dataset['title'].apply(functools.partial(tokenize_title,
                                                                                      window_size=window_size)),
    ).drop(['title', 'first'], axis=1
    ).dropna(axis=0
    )[['x_tokens', 't_tokens']]
    
    dataset.reset_index(inplace=True, drop=True)
    
    dataset = dataset.applymap(functools.partial(labeling, dictionary=dictionary)
    ).assign(x_length=lambda dataset: dataset['x_tokens'].apply(lambda x: len(x)),
             t_length=lambda dataset: dataset['t_tokens'].apply(lambda x: len(x)-window_size-1)
    ).rename(columns={'x_tokens': 'x_labels', 't_tokens': 'yc_and_t_labels'})

    dataset = filtering(dataset)

    dataset = dataset.sort_values('x_length')
    dataset.reset_index(inplace=True, drop=True)

    train_dataset, test_dataset = split_train_test(dataset)
    
    return train_dataset, test_dataset


def split_yc_t(arr, window_size):
    if isinstance(arr[0], str) and isinstance(arr[1], str):
        x_labels = list(map(int, arr[0].replace('[', '').replace(']', '').split(',')))
        yc_and_t_labels = list(map(int, arr[1].replace('[', '').replace(']', '').split(',')))
    else:
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
    
    return result.values

    
def pad_x(x_labels, x_max_length, symbol_ids):
    return x_labels + [symbol_ids['<EOS>']]*(x_max_length-len(x_labels))

def make_p(x_length, x_max_length):
    return [1.0/x_length]*x_length + [0.0]*(x_max_length-x_length)                                     

def make_batch(dataset, symbol_ids, params):
    batch_size = params.batch_size
    window_size =params.window_size
    
    print('splitting yc t')
    pool = Pool(8)
    pool_results = pool.map(functools.partial(split_yc_t,
                                             window_size=window_size), dataset[['x_labels', 'yc_and_t_labels']].values)
    pool.close()
    print('done')
    
    for j, result in enumerate(pool_results):
        sys.stdout.write('\r   j: %10d (%10d)'%(j+1, len(pool_results)))
        if j == 0:
            results = result
        else:
            results = np.r_[results, result] 

    print()
    print(results.shape)
            
    dataset = pd.DataFrame(results, columns=['x_labels', 'yc_labels', 't_label'])
    print('dataset size: ', dataset.shape[0])
    
    dataset = dataset.assign(x_length=lambda dataset: dataset['x_labels'].apply(lambda x: len(x)))
    
    n_batch = int((dataset.shape[0]-1)/batch_size+1)-1 ################### -1 # 最後のやつ捨てます #とりあえず

    print('slicing and padding')
    for i in range(n_batch):
        batch_df = dataset.iloc[i*batch_size: (i+1)*batch_size]
        x_max_length = batch_df['x_length'].max()

        pool = Pool(8)
        pool_results = pool.map(functools.partial(pad_x,
                                                  x_max_length=x_max_length,
                                                  symbol_ids=symbol_ids), batch_df['x_labels'].values)
        pool.close()
        results = []
        for j, result in enumerate(pool_results):
            results.append(result)
        
        series_x = pd.Series(results, name='x_labels')
        batch_df.reset_index(inplace=True, drop=True)
        series_x.reset_index(inplace=True, drop=True)
        batch_df = pd.concat((series_x, batch_df[['yc_labels', 't_label']]), axis=1)
        
        yield i, n_batch, batch_df

def str2list(dataset):
    
    def func(series):
        x_labels = list(map(int, series['x_labels'].replace('[', '').replace(']', '').split(',')))
        yc_and_t_labels = list(map(int, series['yc_and_t_labels'].replace('[', '').replace(']', '').split(','))) 
        return pd.Series([x_labels, yc_and_t_labels], index=['x_labels', 'yc_and_t_labels'])
    
    dataset = dataset.apply(func, axis=1)
    return dataset
    
def save_dataset(path, dataset):
    print('saving dataset...')
    dataset.to_csv(path, index_label=False)

def load_dataset(path, skiprows, nrows):
    col_names = ['x_labels', 'yc_and_t_labels', 't_length', 'x_length']
    with open(path, 'r') as f_load:
        if nrows > 0:
            dataset = pd.read_csv(f_load, skiprows=skiprows, nrows=nrows, names=col_names)
        else:
            dataset = pd.read_csv(f_load, skiprows=skiprows, names=col_names)
    return dataset

def save_dictionary(path, dictionary):
    print('saving dictionary...')
    with open(path, 'wb') as f_dict:
        pickle.dump(dictionary, f_dict)

def load_dictionary(path):
    print('loading dictionary...')
    with open(path, 'rb') as f_dict:
        return pickle.load(f_dict)

def save_batch(path, batch):
    print('saving batch...')
    with open(path, 'wb') as f_batch:
        pickle.dump(batch, f_batch)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--reuters_paths_template', type=str)
    parser.add_argument('--from_db_path', type=str)
    parser.add_argument('--dictionary_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()
    
    pd.set_option('display.width', 10000)

    ### reuters dataset ###
    train_from_reuters = ['Business', 'Domestic', 'Life', 'Oddly_Enough', 'Sports', 'Technology', 'World']
    data_paths = glob.glob(args.reuters_paths_template.replace('\\', ''))
    reuters_for_model = []
    for path in data_paths:
        for category in train_from_reuters:
            if category in path:
                reuters_for_model.append(path)
                break
    print('use file = ', len(reuters_for_model))
    df_reuters = reuters.get(reuters_for_model)
 
    ### from database dataset ###
    df_from_db = from_db.get(args.from_db_path)
  
    ### concatenate ###
    df_all = pd.concat((df_reuters, df_from_db), axis=0)

    with open(args.dictionary_dir+'/alldata.dict', 'rb') as f_dict:
        dictionary = pickle.load(f_dict)
    print('vocabulary size:', len(list(dictionary.keys())))
        
    train_dataset, test_dataset = arrange(df_all, config.params, dictionary)

    print('train size: ', train_dataset.shape[0])
    print('test size: ', test_dataset.shape[0])
    
    save_dataset(args.save_dir+'/train.csv', train_dataset)
    save_dataset(args.save_dir+'/test.csv', test_dataset)
    save_dictionary(args.save_dir+'/dictionary.pkl', dictionary)

    
    # train_dataset = load_dataset(args.save_dir+'/train.csv', 1, -1)
    # with open(args.dictionary_dir+'/alldata.dict', 'rb') as f_dict:
    #     dictionary = pickle.load(f_dict)
    # train_dataset = train_dataset[:1000] #############################################

    
    ### make batch ###
    symbol_ids = {'<S>': dictionary['<S>'], '<EOS>': dictionary['<EOS>']}
    list_batch = list(map(lambda x: x[-1], list(make_batch(train_dataset, symbol_ids, config.params))))
    save_batch(args.save_dir+'/batch.pkl', list_batch)
