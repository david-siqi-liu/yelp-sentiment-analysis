"""
build_features
"""

from sklearn.feature_extraction.text import CountVectorizer
from nltk import download
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


def get_regexp_tokenizer(regexp=r'[a-zA-Z0-9]+') -> RegexpTokenizer:
    """Returns a regular expression tokenizer

    :param regexp: regular expression sequence
    :return: RegexpTokenizer
    """

    return RegexpTokenizer(regexp)


def get_stop_words(language='english') -> list:
    """Returns a list of stop words

    :param language: language, english by default
    :return: list of stop words
    """
    download('stopwords')

    return stopwords.words(language)


def get_count_vectorizer(words, tokenizer=None, stop_words=None, ngram_range=(1, 1)) -> CountVectorizer:
    """Returns a CountVectorizer using all words in the given list

    :param words: list of words
    :param stop_words: list of stop words
    :param ngram_range:
    :param tokenizer:
    :return: CountVectorizer
    """
    if tokenizer:
        cv = CountVectorizer(tokenizer=tokenizer.tokenize, stop_words=stop_words, ngram_range=ngram_range)
    else:
        cv = CountVectorizer(stop_words=stop_words, ngram_range=ngram_range)

    cv.fit(words)
    return cv


def get_examples(phrase, dtm, cv, n=5) -> list:
    """

    :param n:
    :param phrase:
    :param dtm:
    :param cv:
    :return:
    """
    idx = cv.get_feature_names().index(phrase)
    examples = []

    for i in range(dtm.shape[0]):
        if dtm[i].toarray()[0][idx] == 1:
            examples.append(i)
            if len(examples) > n:
                break

    return examples
