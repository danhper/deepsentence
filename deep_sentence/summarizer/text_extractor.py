import requests
from deep_sentence import settings


def make_request(url):
    return requests.get(settings.HTML_EXTRACTOR_URL,
                        params={'url': url},
                        auth=settings.HTML_EXTRACTOR_CREDENTIALS)


# XXX: this should make requests concurrently, but grequests seems to conflict with flask
def extract_from_urls(urls):
    texts = []
    for url in urls:
        response = make_request(url)
        if response.status_code != 200:
            msg = 'could not fetch {0}, got status {1}'.format(response.url, response.status_code)
            raise RuntimeError(msg)
        texts.append(response.text)
    return texts
