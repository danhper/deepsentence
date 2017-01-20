import logging

from . import settings

logger = logging.Logger(settings.APP_NAME)

ch = logging.StreamHandler()

if settings.ENV == 'dev':
    ch.setLevel(logging.DEBUG)
else:
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

logger.addHandler(ch)
