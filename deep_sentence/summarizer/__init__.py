from __future__ import unicode_literals

from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.stemmers import Stemmer
# from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.summarizers.lex_rank import LexRankSummarizer as Summarizer
from sumy.utils import get_stop_words

from deep_sentence import settings

from . import text_extractor


try:
    unicode
except NameError:
    unicode = str


def summarize_text(text, sentences_count=3, language=settings.DEFAULT_LANGUAGE, as_list=False):
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    stemmer = Stemmer(language)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(language)
    sentences = [unicode(sentence) for sentence in summarizer(parser.document, sentences_count)]
    return sentences if as_list else '\n'.join(sentences)


def summarize_texts(texts, sentences_count=3, language=settings.DEFAULT_LANGUAGE):
    options = {'sentences_count': sentences_count, 'language': language, 'as_list': True}
    sentences = [sentence for text in texts for sentence in summarize_text(text, **options)]
    # TODO: deduplicate redundant sentences
    return '\n'.join(sentences)


def summarize_url(url, sentences_count=3, language=settings.DEFAULT_LANGUAGE):
    [text] = text_extractor.extract_from_urls([url])
    return summarize_text(text, sentences_count=sentences_count, language=language)


def summarize_urls(urls, sentences_count=3, language=settings.DEFAULT_LANGUAGE):
    texts = text_extractor.extract_from_urls(urls)
    return summarize_texts(texts, sentences_count=sentences_count, language=language)
