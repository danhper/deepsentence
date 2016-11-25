from setuptools import setup, find_packages

setup(
    name='deep_sentence',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'scrapy': ['settings = deep_sentence.scraper.settings']
    },
    install_requires=[
        'scrapy',
        'python-dotenv',
        'SQLAlchemy',
        'sqlalchemy-migrate',
        'psycopg2',
    ]
)
