from scrapy.exceptions import DropItem

from deep_sentence.scraper import items
from deep_sentence import db, models


class PostgresPipeline(object):
    def __init__(self):
        self.make_session = None
        self.cached_categories = {}
        self.cached_services = {}

    def open_spider(self, _spider):
        self.make_session = db.create_session_maker()

    def process_item(self, item, _spider):
        if isinstance(item, items.CategoryItem):
            self.process_category(item)
        elif isinstance(item, items.ArticleItem):
            self.process_article(item)
        return item

    def process_category(self, category_item):
        if category_item['name'] in self.cached_categories:
            return

        with db.session_scope(self.make_session) as session:
            category = self.create_category(category_item, session)
            self.cached_categories[category.name] = category

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

        sources = [models.Source(**source) for source in article_item.pop('sources')]

        article = models.Article(**article_item)
        article.service = service
        article.category = category
        for source in sources:
            if source.url:
                article.sources.append(source)

        session.add(article)

    def find_service(self, service_name, session):
        if service_name in self.cached_services:
            return self.cached_services[service_name]
        return session.query(models.Service).filter_by(name=service_name).first()

    def find_or_create_category(self, category_name, service, session):
        category = self.find_category(category_name, service.id, session)
        if category:
            return category
        category = models.Category(name=category_name, service=service)
        session.add(category)
        return category

    def find_category(self, category_name, service_id, session):
        if category_name in self.cached_categories:
            return self.cached_categories[category_name]
        query = {'name': category_name, 'service_id': service_id}
        return session.query(models.Category).filter_by(**query).first()
