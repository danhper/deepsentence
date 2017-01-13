from contextlib import contextmanager

import sqlalchemy
from sqlalchemy.orm import sessionmaker

from . import settings


def create_engine(**kwargs):
    return sqlalchemy.create_engine(settings.DATABASE_URL, **kwargs)


def create_session_maker(engine_options=None):
    if not engine_options:
        engine_options = {}
    engine = create_engine(**engine_options)
    return sessionmaker(bind=engine)


@contextmanager
def session_scope(make_session):
    session = make_session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()
