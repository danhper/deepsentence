# Deep Sentence

Deep Sentence is a deep learning based engine to summarize texts from multiple
sources into a single short summary.

## Table of contents

* [Setup](#setup)
* [Scraper](#scraper)
* [Development](#development)
* [DB Setup](#db-setup)

## Setup

Requirements:

* Python 3.5


Setup a new virtualenv environment if you want, then simply run

```
make
```

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

Add the following to your `~/.ssh/config` file

```
Host orion.claudetech.com
    User deep_sentence
    Hostname orion.claudetech.com
    Port 2211
    ForwardAgent yes

Host db.claudetech.com
    User deep_sentence
    ProxyCommand ssh -q orion.claudetech.com nc -q0 192.168.10.39 22
```

Then, start a tunnel to db.claudetech.com:

```
ssh -L 5433:localhost:5432 db.claudetech.com
```

You can then access the database as follow

```
psql -h localhost -p 5433 -U deep_sentence
```
