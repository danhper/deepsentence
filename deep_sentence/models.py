from __future__ import unicode_literals

from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse


Base = declarative_base()

class Service(Base):
    __tablename__ = 'services'
    id = Column(Integer, primary_key=True)
    name = Column(String(255))

    articles = relationship('Article')
    categories = relationship('Category')

    def __repr__(self):
        template = '<Service(id="{0}", name="{1}")>'
        return template.format(self.id, self.name).encode('utf-8')


class Category(Base):
    __tablename__ = 'categories'

    id = Column(Integer, primary_key=True)
    remote_id = Column(String(40))
    name = Column(String(40))
    label = Column(String(255))

    service_id = Column(Integer, ForeignKey('services.id'))
    service = relationship('Service', back_populates='categories')

    articles = relationship('Article')

    def __repr__(self):
        template = '<Category(id="{0}", name="{1}")>'
        return template.format(self.id, self.name).encode('utf-8')


class Article(Base):
    __tablename__ = 'articles'

    id = Column(Integer, primary_key=True)
    remote_id = Column(String(40))
    title = Column(String(255))
    url = Column(String(255))
    content = Column(Text())
    posted_at = Column(DateTime())
    sources_count = Column(Integer())

    service_id = Column(Integer, ForeignKey('services.id'))
    service = relationship('Service', back_populates='articles')

    category_id = Column(Integer, ForeignKey('categories.id'))
    category = relationship('Category', back_populates='articles')

    sources = relationship('Source')

    def __repr__(self):
        template = '<Article(id="{0}", url="{1}", title="{2}")>'
        return template.format(self.id, self.url, self.title).encode('utf-8')


class Source(Base):
    __tablename__ = 'sources'

    id = Column(Integer, primary_key=True)
    title = Column(String(255))
    url = Column(String(255))
    content = Column(Text())
    posted_at = Column(DateTime())

    article_id = Column(Integer, ForeignKey('articles.id'))
    article = relationship('Article', back_populates='sources')

    media_id = Column(Integer, ForeignKey('medias.id'))
    media = relationship('Media', back_populates='sources')

    def __repr__(self):
        template = '<Source(id="{0}", url="{1}", title="{2}")>'
        return template.format(self.id, self.url, self.title).encode('utf-8')

class Media(Base):
    __tablename__ = 'medias'

    id = Column(Integer, primary_key=True)
    base_url = Column(String(255))
    sources_count = Column(Integer())

    sources = relationship('Source')

    def __repr__(self):
        template = '<Media(id="{0}", base_url="{1}")>'
        return template.format(self.id, self.base_url).encode('utf-8')

    @staticmethod
    def extract_base_url(url):
        return urlparse(url).netloc
