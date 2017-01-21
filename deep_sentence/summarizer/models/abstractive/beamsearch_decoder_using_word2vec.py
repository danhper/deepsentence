# -*- coding: utf-8 -*-

from os import path

from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import tensorflow as tf

from deep_sentence.logger import logger
from deep_sentence import settings

from .models_no_embedding_layer import ABSmodel
from . import dataset, config, id2vector

MAX_OUTPUT_LENGTH = 15
BEAM_WIDTH = 3
INPUT_WEIGHT_BEFORE_SOFTMAX = 0.0

pd.set_option('display.width', 10000)

# model_path = '../result_using_word2vec/models/epoch3-batch5000/model.ckpt'
# model_path = '../result_using_word2vec/models/epoch7-batch10000/model.ckpt'
model_path = path.join(settings.MODELS_PATH, 'abstractive/trained/epoch15-batch10000/model.ckpt')

dataset_path = path.join(settings.MODELS_PATH, 'abstractive/data/train.csv')
w2v_path = path.join(settings.MODELS_PATH, 'entity_vector/entity_vector.model.bin')

dictionary_path = path.join(settings.MODELS_PATH, 'abstractive/data/dictionary.pkl')
token2id = dataset.load_dictionary(dictionary_path)
id2token = {i:t for t, i in token2id.items()}

symbol_ids = {'<S>': token2id['<S>'], '<EOS>': token2id['<EOS>']}
vocab_size = len(list(token2id.keys()))
config.params.vocab_size = vocab_size

logger.info('loading dataset...')
dataset_content = dataset.str2list(dataset.load_dataset(dataset_path, 1, 100))

logger.info('loading word2vec model...')
w2v_model = Word2Vec.load_word2vec_format(w2v_path, binary=True)
id_vec_dic = id2vector.make_id_vector_dic(w2v_model, id2token, vocab_size)
del w2v_model

logger.info('setup graph')
sess = tf.Session()
if settings.GPU_NUMBER is not None:
    with tf.device('/gpu:%d' % settings.GPU_NUMBER):
        model = ABSmodel(config.params)
        model.rebuild_forward_graph(sess, model_path)
else:
    model = ABSmodel(config.params)
    model.rebuild_forward_graph(sess, model_path)


def compute_title_from_row(row):
    x = np.array(row[1]['x_labels']).astype(np.int32)
    t = np.array(row[1]['yc_and_t_labels']).astype(np.int32)
    x_words = ' '.join([id2token[a] for a in x])
    t_words = ' '.join([id2token[a] for a in t])


    logger.debug('x words: %s', x_words)
    logger.debug('t words: %s', t_words)
    logger.debug('---------------------------------')

    return compute_title(x)


def transform_input(x):
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return x
    tokens = dataset.tokenize(x)
    return [token2id[token] for token in tokens]

def compute_title(raw_x):
    x = transform_input(raw_x)

    x_vector = np.array([id_vec_dic[a] for a in x])[np.newaxis, :, :]

    output = [np.array([token2id['<S>']]*config.params.window_size)]*BEAM_WIDTH
    output_prob = [1]*BEAM_WIDTH

    x_weight = np.zeros(vocab_size)
    x_weight[x] = INPUT_WEIGHT_BEFORE_SOFTMAX

    for i in range(MAX_OUTPUT_LENGTH):
        temp_output = np.array([])
        temp_output_prob = np.array([])
        for j, o in enumerate(output):
            ### model decoder ###
            y_c = o[-3: ]
            y_c_vector = np.array([id_vec_dic[a] for a in y_c])[np.newaxis, :, :].reshape(1, -1)
            prob = np.squeeze(model.decode(sess, x_vector, y_c_vector, x_weight), 0)
            #print('prob', prob.shape)

            ### beam search  ###
            prob_id = np.sort(prob)[-1::-1][: BEAM_WIDTH] * output_prob[j]
            candidate_ids = np.array([np.r_[output[j], np.array([word_id])] for word_id in np.argsort(prob)[-1::-1][: BEAM_WIDTH]])

            temp_output_prob = np.r_[temp_output_prob, prob_id]

            if temp_output.size == 0:
                temp_output = candidate_ids
            else:
                temp_output = np.r_[temp_output, candidate_ids]

            if i == 0:
                break

        choice = np.argsort(temp_output_prob)[-1::-1][: BEAM_WIDTH]
        output = temp_output[choice]
        output_prob = temp_output_prob[choice]

    best_output = output[0]
    o_words = ' '.join(extract_title(best_output))
    return o_words


def extract_title(output):
    start_token, end_token = symbol_ids['<S>'], symbol_ids['<EOS>'],
    words = []
    for token in output:
        if token == start_token:
            continue
        elif token == end_token:
            break
        else:
            words.append(id2token[token])
    return words


def main():
    import time
    start = time.time()
    # print(dataset_content)
    for row in dataset_content.iterrows():
        o_words = compute_title_from_row(row)
        print('output word: ', o_words)

    print(time.time()-start)
    # print(compute_title(''.join('欧州 中央 銀行 （ ＥＣ Ｂ ） による ３ 年 物 資金 供給 オペ の 効果 が  後退 する 中 、 ユーロ 圏 債務 危機 が 再燃 し て おり 、 銀行 の 資金 調達 環境 を 取り巻く 圧力 が 再び 高まっ て いる'.split(' '))))

if __name__ == '__main__':
    main()
