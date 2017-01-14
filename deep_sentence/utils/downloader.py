import wget
import tarfile
import os
from os import path

from deep_sentence import settings


def download_embeddings():
    output_path = path.join(settings.MODELS_PATH, path.basename(settings.WORD_EMBEDDINGS_URL))
    wget.download(settings.WORD_EMBEDDINGS_URL, output_path)
    compressed = tarfile.open(output_path)
    compressed.extractall(settings.MODELS_PATH)
    os.remove(output_path)
