import scrapy


class ArticleItem(scrapy.Item):
    remote_id = scrapy.Field()
    title = scrapy.Field()
    url = scrapy.Field()
    category = scrapy.Field()
    content = scrapy.Field()
    posted_at = scrapy.Field()
    sources = scrapy.Field()
    service_name = scrapy.Field()


class CategoryItem(scrapy.Item):
    name = scrapy.Field()
    label = scrapy.Field()
    remote_id = scrapy.Field()
    service_name = scrapy.Field()


class SourceItem(scrapy.Item):
    id = scrapy.Field()
    title = scrapy.Field()
    url = scrapy.Field()
    content = scrapy.Field()
