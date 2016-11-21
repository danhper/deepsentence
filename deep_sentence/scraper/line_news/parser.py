import re
from bs4 import BeautifulSoup


def extract_articles(raw_xhtml):
    soup = BeautifulSoup(raw_xhtml, 'lxml')
    for article_soup in soup.find_all('doc-content', attrs={'main': True}):
        parsed_article = parse_article(article_soup)
        if parsed_article:
            yield parsed_article


def parse_article(article_soup):
    article_id = extract_id(article_soup)
    if not article_id:
        return False
    return {
        'article_id': article_id,
        'content': extract_text(article_soup),
        'sources': extract_sources(article_soup)
    }


def article_body(article_soup):
    return article_soup.find(class_=re.compile('Body'))


def extract_sources(article_soup):
    source_soups = article_body(article_soup).find_all(class_=re.compile('CiteTxt'))
    return set(extract_source(source_soup for source_soup in source_soups))


def extract_source(source_soup):
    return {'url': source_soup.attrs['href'], 'title': source_soup.text}


def extract_text(article_soup):
    text_soups = article_body(article_soup).find_all(class_=re.compile('(?<!Cite)Txt'))
    return '\n'.join(text_soup.text for text_soup in text_soups)


def extract_id(article):
    main_image_link = article.find('a', href=re.compile('image_link'))
    if not main_image_link or 'href' not in main_image_link.attrs:
        return
    image_url = main_image_link.attrs['href']
    return image_url.split('/')[-1]
