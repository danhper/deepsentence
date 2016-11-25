try:
    from urllib.parse import urlencode
except ImportError:
    from urllib import urlencode

import time

import scrapy

from . import parser


class LineNewsSpider(scrapy.Spider):
    name = "line_news"
    allowed_domains = ["news.line.me"]
    base_url = 'http://news.line.me/'
    api_base_url = 'http://news.line.me/api/v2'
    api_headers = {'X-From': 'http://news.line.me'}

    def start_requests(self):
        # NOTE: we should add logic to start scraping from a given URL here if needed
        yield scrapy.Request(LineNewsSpider.base_url, callback=self.parse_categories)

    def parse_categories(self, response):
        for category in parser.extract_categories(response):
            yield category
            yield self.make_category_request(category)

    def make_category_request(self, category):
        timestamp = int(time.time() * 1000)
        params = urlencode({'category_id': category['remote_id'], '_': timestamp})
        return scrapy.Request(LineNewsSpider.api_base_url + '/issue/top?' + params,
                              headers=LineNewsSpider.api_headers,
                              callback=self.parse_api_response)

    def parse_api_response(self, response):
        for url in parser.extract_urls_from_api_response(response):
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        yield parser.extract_article(response)
        for url in parser.extract_related_urls(response):
            yield scrapy.Request(url=url, callback=self.parse)
