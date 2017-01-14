import inspect
import os
from os import path

from dotenv import load_dotenv


def try_load_dotenv():
    dot_env_path = path.join(PROJECT_ROOT, '.env')
    if path.isfile(dot_env_path):
        load_dotenv(dot_env_path)


PROJECT_ROOT = os.environ.get('PROJECT_ROOT',
                              path.dirname(path.dirname(inspect.getfile(inspect.currentframe()))))

try_load_dotenv()

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgres://localhost/deep_sentence_dev')
HTML_EXTRACTOR_BASE_URL = os.environ.get('HTML_EXTRACTOR_BASE_URL',
                                         'http://extractor.deepsentence.com')
HTML_EXTRACTOR_URL = path.join(HTML_EXTRACTOR_BASE_URL, 'extract')

HTML_EXTRACTOR_CREDENTIALS = (
    os.environ.get('HTML_EXTRACTOR_USER', 'deep_sentence'),
    os.environ.get('HTML_EXTRACTOR_PASSWORD', ''),
)


FIXTURES_PATH = path.join(PROJECT_ROOT, 'deep_sentence', 'fixtures')

WORD_EMBEDDINGS_URL = 'http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/entity_vector.tar.bz2'

MODELS_PATH = os.environ.get('MODELS_PATH', path.join(PROJECT_ROOT, 'models'))
