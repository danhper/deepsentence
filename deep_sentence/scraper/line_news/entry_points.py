import json
from datetime import datetime

import grequests

from . import constants
from .categories import fecth_categories


API_ENDPOINT = constants.LINE_NEWS_API_URL + '/issue/top'
ENCODING = 'utf-8-sig'


def fetch_entry_points():
    categories = fecth_categories()
    reqs = [make_request(category) for category in categories]
    responses = grequests.map(reqs)
    entry_points = [fetch_entry_point(response) for response in responses if response]
    return [entry_point for entry_point in entry_points if entry_point]


def make_request(category):
    timestamp = int(datetime.timestamp(datetime.now()) * 1000)
    params = {'category_id': category['remote_id'], '_': timestamp}
    return grequests.get(API_ENDPOINT, params=params, headers=constants.DEFAULT_API_HEADERS)


def fetch_entry_point(response):
    try:
        body = json.loads(response.content.decode(ENCODING))
        return body['result']['topic']['issues'][0]['item']['page_url']
    except (KeyError, json.JSONDecodeError):
        return False
