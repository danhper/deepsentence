import inspect
import os
from os import path

from dotenv import load_dotenv


def try_load_dotenv():
    dot_env_path = path.join(PROJECT_ROOT, '.env')
    if path.isfile(dot_env_path):
        load_dotenv(dot_env_path)


PROJECT_ROOT = os.environ.get('PROJECT_ROOT', path.dirname(path.dirname(inspect.getfile(inspect.currentframe()))))

try_load_dotenv()

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgres://localhost/deep_sentence_dev')

FIXTURES_PATH = path.join(PROJECT_ROOT, 'deep_sentence', 'fixtures')
