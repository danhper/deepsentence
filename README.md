# Deep Sentence

Deep Sentence is a deep learning based engine to summarize texts from multiple
sources into a single short summary.

## Table of contents

* [Setup](#setup)
* [Scraper](#scraper)
* [Development](#development)
* [DB Setup](#db-setup)
* [Deployment](#deployment)

## Setup

### Requirements

* Python 3.5
* [psycopg2 requirements](http://initd.org/psycopg/docs/install.html)

### Installing dependencies

Setup a new virtualenv environment if you want, then simply run

```
make
```

### Configuration

Copy `.env.example` to `.env`, and modify the variables to your needs.

## Scraper

### Usage

To start the scraper, run

```
scrapy crawl line_news
```

if you want a shell to play around with the responses, run

```
scrapy shell ARTICLE_URL --spider=line_news
```

## Development

### Adding dependencies

Run

```
make write_dependencies
```

to regenerate `requirements.txt`.
Please be sure to run this from a clean environment, and only add *needed* dependencies.

## DB setup

You can access the database as follow

```
psql -h public-db.claudetech.com -p 5433 -U deep_sentence
```

To be able to use it in from Python, set `DATABASE_URL` to the following value

```
postgres://deep_sentence:PASSWORD@public-db.claudetech.com:5433/deep_sentence
```

## Deployment

See (./deployment/README.md) for more information about how to setup a node.
