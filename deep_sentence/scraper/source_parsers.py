import sys

def find_parser_for(url):
    module = sys.modules[__name__]
    for attr in dir(module):
        if not attr.endswith('Parser'):
            continue
        klass = getattr(module, attr)
        if url in klass.supported_urls:
            return klass()


class BaseParser(object):
    supported_urls = []

    def extract_content(self, response):
        raise NotImplementedError('extract_content must be implemented')


class JijiParser(BaseParser):
    supported_urls = ['www.jiji.com']

    def extract_content(self, response):
        article_block = response.xpath('//*[contains(@class, "ArticleBlock")]')
        if not article_block:
            return
        content = article_block[0].xpath('./*[contains(@class, "ArticleText")]/text()').extract()
        return '\n'.join(content)
