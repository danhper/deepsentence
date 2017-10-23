import argparse
import numpy as np
import pandas as pd
import MeCab
from deep_sentence.summarizer.utils import w2v

import tinysegmenter

WORD_CLASS = ['名詞', '動詞', '形容詞', '形容動詞', '副詞']


def tiny_tokenize(sentence):
    tokenizer = tinysegmenter.TinySegmenter()
    return tokenizer.tokenize(sentence)


def tokenize_POSfilter(sentence):
    try:
        tagger = MeCab.Tagger('-Ochasen')
        tagger.parse('')
        node = tagger.parseToNode(sentence)
        filtered = []
        while node:
            if str(node.feature.split(',')[0]) in WORD_CLASS:
                filtered.append(node.surface)
            node = node.next
        return filtered
    except RuntimeError:
        return tiny_tokenize(sentence)


def tokenize(sentence):
    try:
        mt    = MeCab.Tagger('-Owakati')
        parse = mt.parse(sentence)
        return parse.split()
    except UnicodeDecodeError:
        return tiny_tokenize(sentence)


def word_similarity(word1, word2):
    try:
        return w2v.model.similarity(word1, word2)
    except (KeyError, ValueError):
        return 0.0

def sentences_similarity(sentence1, sentence2):
    '''
    sentence1: 要約文
    sentence2: 元記事
    '''
    tokens1 = tokenize_POSfilter(sentence1)
    tokens2 = tokenize_POSfilter(sentence2)

    w_scores = []
    for word1 in tokens1:
        w_score = 0.0
        for word2 in tokens2:
            temp = word_similarity(word1, word2)
            if temp > w_score:
                w_score = temp
                if temp == 1.0:
                    break
        w_scores.append(w_score)
    s_score = np.average(np.array(w_scores))

    return s_score


def main():
    pd.set_option('display.width', 1000)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_path', type=int, default=None, help='t')
    args = parser.parse_args()

    summary = '年末年始を海外や古里で過ごす人たちの移動ラッシュがピークを迎え、空港やターミナル駅は29日午前、大きなキャリーバッグを引く旅行者や家族連れらでにぎわいました。'
    sentence1 = '年末年始を海外や古里で過ごす人たちの移動ラッシュがピークを迎え、空港やターミナル駅は29日午前、大きなキャリーバッグを引く旅行者や家族連れらでにぎわった。'
    sentence2 = '成田空港では、妻と一緒にハワイで年越しするというさいたま市の会社員関根彦一さんが「きれいな風景を写真に収めたい」と話した。成田国際空港会社によると、29日の出国者数は約5万人の見込み。来年1月3日が帰国ピークとしている。'
    sentence3 = 'JR東京駅の新幹線ホームでは帰省客が目立った。宮城県の実家に帰省する東京都の主婦、水野一美さんは「7カ月の長男の顔を両親に見せてあげたい。私は少しのんびり過ごせたらいいな」と話した。'

    print(sentence_similarity(summary, sentence1))
    print(sentence_similarity(summary, sentence2))
    print(sentence_similarity(summary, sentence3))


if __name__ == '__main__':
    main()
