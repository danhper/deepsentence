import sys

def find_parser_for(url):
    module = sys.modules[__name__]
    for attr in dir(module):
        if not attr.endswith('Parser'):
            continue
        klass = getattr(module, attr)
        if url in klass.supported_urls:
            return klass


class BaseParser(object):
    supported_urls = []

    def __init__(self, response):
        self.response = response

    def extract_content(self):
        raise NotImplementedError('extract_content must be implemented')

    def extract_texts(self, selector):
        texts = self.response.xpath(selector).extract()
        return '\n'.join(self.strip_and_filter_texts(texts))

    def strip_texts(self, texts):
        return [text.strip() for text in texts]

    def strip_and_filter_texts(self, texts):
        return [text for text in self.strip_texts(texts) if text]


class JijiParser(BaseParser):
    supported_urls = ['www.jiji.com']

    def extract_content(self):
        article_block = self.response.xpath('//*[contains(@class, "ArticleBlock")]')
        if not article_block:
            return
        content = article_block[0].xpath('./*[contains(@class, "ArticleText")]/text()').extract()
        return '\n'.join(content)

class OriconParser(BaseParser):
    supported_urls = ['www.oricon.co.jp']

    def extract_content(self):
        selector = '//*[contains(@class, "content")]' + \
                   '//*[contains(@class,"box-a")]' + \
                   '//*[self::p or self::span]//text()'
        return self.extract_texts(selector)


class SanspoParser(BaseParser):
    supported_urls = ['www.sanspo.com']

    def extract_content(self):
        return self.extract_texts('//*[contains(@class, "NewsDetail")]//p/text()')

class ITMediaParser(BaseParser):
    supported_urls = ['nlab.itmedia.co.jp']

    def extract_content(self):
        return self.extract_texts('//*[@id="cmsBody"]//p/text()')
