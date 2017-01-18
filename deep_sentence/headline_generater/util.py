import numpy as np

import sys
sys.path.append('../')

from models import *
from setting import *

import sqlalchemy
from sqlalchemy.orm import sessionmaker

import MeCab


def create_engine():
    return sqlalchemy.create_engine(DATABASE_URL)


def create_session_maker():
    engine = create_engine()
    return sessionmaker(bind=engine)


def load_embeddings():
  dic = {}
  vec = []
  with open(EMBEDDING_FILE) as f:
    reader = f.readlines()[1:]
    for i, row in enumerate(reader):
      row = row.replace('\n','').split(' ')
      dic[i] = row[0] if not row[0].startswith('[') else row[0][1:-1]
      vec.append(np.array(row[1:],dtype=np.float32))
  vec = np.array(vec)
  return dic, vec


def make_dataset():
  # Load dataset from psql.
  session_maker = create_session_maker()
  session = session_maker()
  articles = session.query(Article).order_by(Article.id).slice(0,1000).all()

  # Load embeddings.
  dic, vec = load_embeddings()
  word2id = {v:k for k, v in dic.items()}

  # Transform dataset into source_ids and target_ids.
  tagger = MeCab.Tagger('-Owakati')
  sources = [tagger.parse(article.content.replace('\n','').replace(' ',''))[:-2].split(' ') for article in articles]
  targets = [tagger.parse(article.title.replace('\n','').replace(' ',''))[:-2].split(' ') for article in articles]

  def func(x):
    try:
      return word2id[x]
    except KeyError:
      return "@"

  source_ids = [[func(x)for x in source] for source in sources]
  target_ids = [[func(x)for x in target]+['EOS'] for target in targets]
  return source_ids, target_ids, dic, vec
