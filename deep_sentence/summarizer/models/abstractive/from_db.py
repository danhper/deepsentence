# -*- coding: utf-8 -*-

import re
import numpy as np
import pandas as pd


def load_rawdata(path):
    data = pd.read_pickle(path).dropna(axis=0)[['title', 'content']]
    return data

def load_rawdata(path):
    data = pd.read_pickle(path).dropna(axis=0)[['title', 'content']]
    return data

def first_sentence(sentences):
    sentences = sentences.replace('\n', '')
    try:
        first = re.match('^(.*?)。', sentences).group(0)
    except AttributeError:
        return None
    return first

def remove_from_title(title):
    use_media = ['産経ニュース', 'MSN産経west', '産経WEST', 'スポニチ', 'デイリースポーツ', 'サンスポ', '共同通信', 'シネマトゥデイ', 'シネマカフェ', 'モデルプレス', 'ゲキサカ', '東スポ', 'スポーツ報知', '日刊スポーツ', '時事ドットコム', 'ファッションプレス', 'ORICON STYLE', 'ナタリー', 'J-CASTニュース', '47NEWS', 'Glitty', 'エキサイトコネタ', 'GIGAZINE', 'ねとらぼ', 'TechCrunch', 'MSN産経ニュース', 'livedoorニュース', 'ウォーカープラス', '音楽ナタリー', 'ZAKZAK', 'SANSPO.COM', 'Forbes JAPAN', 'KYODO NEWS', '映画.com', 'CINRA.NET', 'アニメ！アニメ！', 'BIGLOBEニュース']
    m = re.match('^(.*?)(%s)'%('|'.join(use_media)), title)
    if m:
        return m.group(1)
    else:
        return None

def get(path):
    data = load_rawdata(path)
    dataset = data.assign(first=lambda data: data['content'].apply(first_sentence),
                          title_=lambda data: data['title'].apply(remove_from_title)
    ).drop(['title', 'content'], axis=1
    ).dropna(axis=0
    ).rename(columns={'title_': 'title'}
    )[['title', 'first']]
    dataset.reset_index(inplace=True, drop=True)

    return dataset
    
if __name__ == '__main__':


    get('../data/raw_data/from_db/sources.pkl')
    
