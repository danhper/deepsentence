import os
from os import path

import tarfile
import wget

from deep_sentence import settings


def download_tar(url, output_dir):
    output_path = path.join(output_dir, path.basename(url))
    wget.download(url, output_path)
    compressed = tarfile.open(output_path)
    compressed.extractall(output_dir)
    os.remove(output_path)


def download_embeddings():
    download_tar(settings.WORD_EMBEDDINGS_URL, settings.MODELS_PATH)


def download_abstractive_data():
    download_tar(settings.ABSTRACTIVE_DATA_URL, path.join(settings.MODELS_PATH, 'abstractive'))


def download_abstractive_trained():
    download_tar(settings.ABSTRACTIVE_TRAINED_URL, path.join(settings.MODELS_PATH, 'abstractive'))
