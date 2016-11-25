import json
import re
from datetime import datetime
try:
    from urllib.parse import urlparse, urlunparse
except ImportError:
    from urlparse import urlparse, urlunparse

from deep_sentence.scraper.items import ArticleItem, CategoryItem, SourceItem


SERVICE_NAME = 'line_news'
SELECTORS = {
    'title': '//h2[contains(@class, "Ttl")]/text()',
    'body': '//*[contains(@class, "Contents")]/*[contains(@class, "Txt") and not (contains(@class, "CiteTxt"))]/text()',
    'date': '//*[contains(@class, "Date")]/text()',
    'sources': '//*[contains(@class, "Contents")]//*[contains(@class, "CiteWrap")]',
    'source_url': './/a[contains(@class, "CiteTxt")]/@href',
    'related_urls': '//doc-viewer/@src',
    'navigation_elems': '//nav[@role="navigation"]//li'
}


def parse_date(date):
    no_year_regexp = r'\d{1,2}.\d{1,2} \d{1,2}:\d{1,2}'  # e.g. 04.20 13:45
    long_year_regexp = r'\d{4}.' + no_year_regexp        # e.g. 2016.04.20 13:45
    short_year_regexp = r'\d{2}' + no_year_regexp        # e.g. 16.04.20 13:45

    if re.match(no_year_regexp, date):
        return datetime.strptime(date, '%m.%d %H:%M').replace(year=datetime.now().year)
    elif re.match(long_year_regexp, date):
        return datetime.strptime(date, '%Y.%m.%d %H:%M')
    elif re.match(short_year_regexp, date):
        return datetime.strptime(date, '%y.%m.%d %H:%M')


def extract_urls_from_api_response(response):
    body = json.loads(response.text)
    return [link['item']['page_url'] for link in body['result']['topic']['issues']]


def extract_categories(response):
    selectors = response.xpath(SELECTORS['navigation_elems'])
    categories = []
    for selector in selectors:
        category = extract_category(selector)
        if category and not any(c['remote_id'] == category['remote_id'] for c in categories):
            categories.append(category)
    return categories


def extract_category(selector):
    try:
        remote_id = int(selector.xpath('@data-category-id').extract_first())
    except ValueError:
        remote_id = -1

    if remote_id <= 0:
        return False

    return CategoryItem(
        name=selector.xpath('@data-category-name').extract_first(),
        label=selector.xpath('./a/text()').extract_first(),
        remote_id=remote_id,
        service_name=SERVICE_NAME
    )


def strip_and_filter_texts(texts):
    return filter(lambda x: x, [text.strip() for text in texts])


def extract_article(response):
    content = '\n'.join(strip_and_filter_texts(response.xpath(SELECTORS['body']).extract()))
    posted_at = parse_date(response.xpath(SELECTORS['date']).extract_first())
    category_match = re.match('http://news.line.me/issue/(.+)/.+', response.url)
    if category_match:
        category = category_match.group(1)
    return ArticleItem(
        remote_id=response.url.split('/')[-1],
        title=response.xpath(SELECTORS['title']).extract_first(),
        category=category,
        url=response.url,
        content=content,
        posted_at=posted_at,
        sources=extract_sources(response),
        service_name=SERVICE_NAME
    )


def extract_sources(response):
    sources = []
    for source_selector in response.xpath(SELECTORS['sources']):
        source = extract_source(source_selector)
        if not any(source['url'] == s['url'] for s in sources):
            sources.append(source)
    return sources


def extract_source(selector):
    texts = selector.xpath('.//text()').extract()
    title = ' '.join(strip_and_filter_texts(texts))
    url = selector.xpath(SELECTORS['source_url']).extract_first()
    return SourceItem(url=url, title=title)


def path_to_url(response, path):
    return urlunparse(urlparse(response.url)._replace(path=path))


def extract_related_urls(response):
    paths = response.xpath(SELECTORS['related_urls']).extract()
    return [path_to_url(response, path) for path in paths]
