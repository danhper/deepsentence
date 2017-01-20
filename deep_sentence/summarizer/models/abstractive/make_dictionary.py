# -*- coding: utf-8 -*-

import argparse
import glob
import pickle
import MeCab
import gensim
import numpy as np
import pandas as pd

import reuters
import from_db

MIN_FREQUENCY = 5

def tokenize(sentence):
    mt    = MeCab.Tagger('-Owakati') 
    parse = mt.parse(sentence)
    return parse.split()

def save(df_all, dictionary, bow, path):
    print('saving...')
    df_all.to_csv(path+'/tokens.csv', index=False)
    with open(path+'/alldata.dict', 'wb') as f_dict:
        pickle.dump(dictionary, f_dict)
    with open(path+'/bow.pkl', 'wb') as f_bow:
        pickle.dump(bow, f_bow)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--reuters_paths_template', type=str)
parser.add_argument('--from_db_path', type=str)
parser.add_argument('--save_dir', type=str)
args = parser.parse_args()

reuters_paths = glob.glob(args.reuters_paths_template.replace('\\', ''))
df_reuters = reuters.load_rawdata(reuters_paths)
df_from_db = from_db.load_rawdata(args.from_db_path)

df_all = pd.concat((df_reuters, df_from_db), axis=0).applymap(tokenize)

dictionary = gensim.corpora.Dictionary(df_all.values.flatten())
dictionary.add_documents([['<S>', '<EOS>', '<UNK>', '<#>']])
vocab_size = len(list(dictionary.iterkeys()))
print(vocab_size)
bow = dict(dictionary.doc2bow([elem for list_elem in df_all.values.flatten() for elem in list_elem]+['<S>', '<EOS>', '<UNK>', '<#>']))

new_dictionary = {'<S>': 0,
                  '<EOS>': 1,
                  '<UNK>': 2,
                  '<#>': 3}

val = 4
print()
for k in dictionary.token2id.keys():
    if bow[dictionary.token2id[k]] >= MIN_FREQUENCY:
        new_dictionary[k] = val
        val += 1

print(len(dictionary.token2id))
print(len(new_dictionary))

save(df_all, new_dictionary, bow, args.save_dir)

