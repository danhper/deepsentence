import os
from os import path

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgres://localhost/deep_sentence_dev')

FIXTURES_PATH = path.join(path.dirname(path.realpath(__file__)), 'fixtures')
