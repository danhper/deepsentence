# -*- coding: utf-8 -*-

import re
import numpy as np
import pandas as pd

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

    return data

def remove_from_title(title):
    remove_list = ['再送:', '再送：', '再送-', '焦点:', '焦点：', '訂正:', '訂正：', '訂正-', 'アングル：', ' ']
    for string in remove_list:
        title = title.replace(string, '')
    return title

def remove_from_first(first):
    return first.replace('()', '')

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
    return remove_from_first(m.group(1))

def filtering(dataset):
    def func(series):
        title = series['title']
        in_list = ['ロイター調査：', 'ロイター調査：' , 'ロイター企業調査:', 'ロイター企業調査：', 'インタビュー:', 'インタビュー：']
        out_list1 = ['サマリー', '情報ＢＯＸ：']
        out_list2 = [':', '：']
        flg_in = np.array([elem in title for elem in in_list]).any()
        flg_out1 = np.array([elem in title for elem in out_list1]).any()
        flg_out2 = np.array([elem in title for elem in out_list2]).any()
            
        if flg_out1:
            return pd.Series([None, None], index=['title', 'first'])
        elif flg_out2 and not flg_in:
            return pd.Series([None, None], index=['title', 'first'])
        else:
            return series
        
    dataset = dataset.apply(func, axis=1).dropna(axis=0)
    return dataset

def get(paths):
    data = load_rawdata(paths)
    dataset = data.assign(first=lambda data: data['content'].apply(first_sentence),
                          title_=lambda data: data
                          ['title'].apply(remove_from_title),
    ).drop(['title', 'content'], axis=1
    ).rename(columns={'title_': 'title'}
    )[['title', 'first']]
    dataset = dataset.dropna(axis=0)
    dataset = filtering(dataset)
    dataset.reset_index(inplace=True, drop=True)

    return dataset

if __name__ == '__main__':
    import glob
    data = get(glob.glob('../data/raw_data/reuters/Business/*/*.csv'))
    
    print(dataset)
