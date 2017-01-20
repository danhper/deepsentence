from os import path

from gensim.models import Word2Vec

from deep_sentence import settings


def load_model():
    model_path = path.join(settings.MODELS_PATH, 'entity_vector/entity_vector.model.bin')
    return Word2Vec.load_word2vec_format(model_path, binary=True)


model = load_model()
