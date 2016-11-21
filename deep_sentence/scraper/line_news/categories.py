import requests
from bs4 import BeautifulSoup

from . import constants

def fecth_categories():
    r = requests.get(constants.LINE_NEWS_URL, headers=constants.DEFAULT_HEADERS)
    return extract_categories(r.text)


def extract_categories(top_page_xhtml):
    soup = BeautifulSoup(top_page_xhtml, 'lxml')
    categories_soup = soup.find('nav', attrs={'role': 'navigation'}).find_all('li')
    result = []
    for category_soup in categories_soup:
        category = extract_category(category_soup)
        if category and not any(c['remote_id'] == category['remote_id'] for c in result):
            result.append(category)
    return result


def extract_category(category_soup):
    try:
        remote_id = int(category_soup.attrs['data-category-id'])
    except ValueError:
        remote_id = -1

    if remote_id <= 0:
        return False

    return {
        'name': category_soup.attrs['data-category-name'],
        'label': category_soup.text,
        'remote_id': remote_id,
        'service_name': constants.SERVICE_NAME
    }
