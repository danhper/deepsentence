import scrapy

from deep_sentence.scraper.source_parsers import find_parser_for
from deep_sentence.scraper.items import SourceItem


class SourceContentSpider(scrapy.Spider):
    name = 'source_content'

    def __init__(self, *args, **kwargs):
        scrapy.Spider.__init__(self, *args, **kwargs)
        self.urls_metadata = kwargs.get('urls_metadata')
        if not self.urls_metadata:
            raise ValueError('you must provide urls metadata')

    def start_requests(self):
        return [scrapy.Request(key, meta=value, callback=self.parse)
                for (key, value) in self.urls_metadata.items()]

    def parse(self, response):
        Parser = find_parser_for(response.meta['base_url'])
        if Parser:
            content = Parser(response).extract_content()
            if not content:
                return
        yield SourceItem(id=response.meta['id'], content=content)
