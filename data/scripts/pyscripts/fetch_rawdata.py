# -*- coding: utf-8 -*-

import argparse
from getpass import getpass
import psycopg2
import numpy as np
import pandas as pd

class fetch():

    def __init__(self, password):
        self.connection = psycopg2.connect('host=public-db.claudetech.com port=5433 user=deep_sentence password=%s' % password)
        self.cursor = self.connection.cursor()
        
    def select(self, col, table, n_rows=None):
        if n_rows:
            self.cursor.execute('SELECT %s FROM %s LIMIT %d' % (col, table, n_rows))
        else:
            self.cursor.execute('SELECT %s FROM %s' % (col, table))

    def get_field(self, table):
        self.select('*', table)
        fields = []
        for item in self.cursor.description:
            fields.append(item[0])
        print(fields)

    def get_articles(self, n_rows=None):
        # filed: 'id', 'remote_id', 'title', 'url', 'service_id', 'category_id', 'content', 'posted_at', 'sources_count'
        colnames = ['id', 'title', 'category_id', 'content'] 
        self.select(', '.join(colnames), 'articles', n_rows)
        self.articles = pd.DataFrame(self.cursor.fetchall(), columns=colnames)

    def get_sources(self, n_rows=None):
        # filed: 'id', 'url', 'title', 'article_id', 'content', 'posted_at', 'media_id'
        colnames = ['id', 'title', 'article_id', 'content']
        self.select(', '.join(colnames), 'sources', n_rows)
        self.sources = pd.DataFrame(self.cursor.fetchall(), columns=colnames)
        
    def get_categories(self):
        # filed: 'id', 'name', 'label', 'service_id', 'remote_id'
        colnames = ['id', 'name', 'label']
        self.select(', '.join(colnames), 'categories')
        self.categories = pd.DataFrame(self.cursor.fetchall(), columns=colnames)

    def locate_sources(self):
        self.correspondence = self.articles.copy()
        n_col_articles = self.articles.shape[1]
        for i, article_id in enumerate(self.sources['article_id']):
            row = np.where(self.correspondence['id']==article_id)[0]
            if len(row):
                column = 'article%d' % (self.correspondence.iloc[row[0]].notnull().sum()-n_col_articles+1)
                self.correspondence.ix[row[0], column] = self.sources.ix[i, 'content']
        self.correspondence = self.correspondence.drop('id', axis=1)
        
    def __del__(self):
        self.cursor.close()
        self.connection.close()

if __name__ == '__main__':

    pd.set_option('display.width', 1000)
    
    parser = argparse.ArgumentParser(description='fetch rawdata from database')
    parser.add_argument('--password', type=str, default=None,
                        help='remove from git management')
    parser.add_argument('--n_rows', type=int, default=None,
                        help='number of row to select')
    parser.add_argument('output_path', type=str, default=10,
                        help='output filename')
    args = parser.parse_args()

    if args.password:
        pw = args.password
    else:
        pw = getpass('database password: ')

    F = fetch(pw)
    F.get_articles(args.n_rows)
    F.get_sources(args.n_rows)
    F.locate_sources()
    
    F.correspondence.to_pickle(args.output_path)
    
   
