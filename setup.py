from setuptools import setup

setup(
    name='deep_sentence',
    version='0.1.0',
    packages=['deep_sentence'],
    install_requires=[
        'scrapy',
        'python-dotenv',
        'SQLAlchemy',
        'sqlalchemy-migrate',
        'psycopg2',
    ]
)
