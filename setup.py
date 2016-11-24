from setuptools import setup

setup(
    name='deep_sentence',
    packages=['deep_sentence'],
    install_requires=[
        'scrapy',
        'python-dotenv',
        'SQLAlchemy',
        'sqlalchemy-migrate',
        'psycopg2',
    ]
)
