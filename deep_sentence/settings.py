import os
from os import path

from dotenv import load_dotenv


def try_load_dotenv():
    dot_env_path = path.join(PROJECT_ROOT, '.env')
    if path.isfile(dot_env_path):
        load_dotenv(dot_env_path)


PROJECT_ROOT = path.dirname(path.dirname(path.realpath(__file__)))

try_load_dotenv()

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgres://localhost/deep_sentence_dev')

FIXTURES_PATH = path.join(path.dirname(path.realpath(__file__)), 'fixtures')
