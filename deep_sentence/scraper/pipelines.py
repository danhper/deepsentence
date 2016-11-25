from scrapy.exceptions import DropItem

from deep_sentence.scraper import items
from deep_sentence import db, models


class PostgresPipeline(object):
    def __init__(self):
        self.make_session = None

    def open_spider(self, _spider):
        self.make_session = db.create_session_maker()

    def process_item(self, item, _spider):
        if isinstance(item, items.CategoryItem):
            self.process_category(item)
        elif isinstance(item, items.ArticleItem):
            self.process_article(item)
        return item

    def process_category(self, category_item):
        with db.session_scope(self.make_session) as session:
            self.create_category(category_item, session)

    def create_category(self, category_item, session):
        service_name = category_item.pop('service_name')
        service = self.find_service(service_name, session)
        if not service:
            raise DropItem('could not find service {0}'.format(service_name))

        category = self.find_category(category_item['name'], service.id, session)
        if category:
            return category

        category = models.Category(**category_item)
        category.service = service
        session.add(category)
        return category

    def process_article(self, article_item):
        with db.session_scope(self.make_session) as session:
            article = session.query(models.Article).filter_by(url=article_item['url']).first()
            if not article:
                self.process_new_article(article_item.copy(), session)

    def process_new_article(self, article_item, session):
        service_name = article_item.pop('service_name')
        service = self.find_service(service_name, session)
        if not service:
            raise DropItem('could not find service {0}'.format(service_name))

        category_name = article_item.pop('category')
        category = self.find_or_create_category(category_name, service, session)
        if not category:
            raise DropItem('could not find category {0}'.format(category_name))

        sources = self.generate_sources(article_item.pop('sources'), session)

        article = models.Article(**article_item)
        article.service = service
        article.category = category
        for source in sources:
            article.sources.append(source)

        article.sources_count = len(article.sources)

        session.add(article)

    def generate_sources(self, source_items, session):
        session.autoflush = False
        sources = []
        for source_item in source_items:
            if source_item['url']:
                source = models.Source(**source_item)
                media = self.find_or_create_media(source.url, session)
                source.media = media
                source.media.sources_count += 1
                sources.append(source)
        session.autoflush = True
        return sources

    def find_or_create_media(self, url, session):
        base_url = models.Media.extract_base_url(url)
        media = session.query(models.Media).filter_by(base_url=base_url).first()
        if media:
            return media
        media = models.Media(base_url=base_url, sources_count=0)
        return media

    def find_service(self, service_name, session):
        return session.query(models.Service).filter_by(name=service_name).first()

    def find_or_create_category(self, category_name, service, session):
        category = self.find_category(category_name, service.id, session)
        if category:
            return category
        category = models.Category(name=category_name, service=service)
        session.add(category)
        return category

    def find_category(self, category_name, service_id, session):
        query = {'name': category_name, 'service_id': service_id}
        return session.query(models.Category).filter_by(**query).first()
