"""
build_features
"""

from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import wordnet
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag


class PreProcessor(BaseEstimator, TransformerMixin):

    def __init__(self, stopwords=None, stemmer=None, lemmatizer=None):
        self.stopwords = stopwords
        self.stemmer = stemmer
        self.lemmatizer = lemmatizer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [
            list(self.tokenize(review)) for review in X
        ]

    def tokenize(self, review):
        # Break the review into sentences
        for sent in sent_tokenize(review):
            # Break the sentence into part-of-speech (POS) tagged tokens
            for token, tag in pos_tag(word_tokenize(sent)):
                # Apply pre-processing to the token

                # Lower-case and strip
                token = token.lower()
                token = token.strip()

                # Stopwords
                if token in self.stopwords:
                    continue

                # Stemming
                if self.stemmer:
                    token = self.stemmer.stem(token)

                # Lemmatization
                if self.lemmatizer:
                    token = self.lemmatize(token, tag)

                yield token

    def lemmatize(self, token, tag):
        tag = {
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV,
            'J': wordnet.ADJ
        }.get(tag[0], wordnet.NOUN)

        return self.lemmatizer.lemmatize(token, tag)


def get_examples(phrase, dtm, cv, n=5) -> list:
    idx = cv.get_feature_names().index(phrase)
    examples = []

    for i in range(dtm.shape[0]):
        if dtm[i].toarray()[0][idx] == 1:
            examples.append(i)
            if len(examples) > n:
                break

    return examples
