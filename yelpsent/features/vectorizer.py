"""
vectorizer
"""
import re
import string

from nltk import PorterStemmer, WordNetLemmatizer, sent_tokenize, wordpunct_tokenize, pos_tag
from nltk.corpus import wordnet, stopwords
from sklearn.feature_extraction.text import CountVectorizer


class YelpSentCountVectorizer(CountVectorizer):
    def __init__(self, ngram_range=(1, 1),
                 remove_nonwords=False, remove_stopwords=False,
                 stem=False, lemmatize=False, min_df=1, binary=False):
        super().__init__()
        self.punct = set(string.punctuation)
        self.ngram_range = ngram_range
        self.remove_nonwords = remove_nonwords
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.stemmer = PorterStemmer() if stem else None
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        self.min_df = min_df
        self.binary = binary

    def lemmatize(self, token, tag):
        tag = {
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV,
            'J': wordnet.ADJ
        }.get(tag[0], wordnet.NOUN)
        return self.lemmatizer.lemmatize(token, tag)

    def stem(self, token):
        return self.stemmer.stem(token)

    def build_analyzer(self):
        # create the analyzer that will be returned by this method
        def analyser(doc):
            # Keep only words
            doc = re.sub('[^A-Za-z0-9]+', ' ', doc) if self.remove_nonwords else doc
            cleaned_tokens = []
            # Break the document into sentences
            for sent in sent_tokenize(doc):
                # Break the sentence into part of speech tagged tokens
                for token, tag in pos_tag(wordpunct_tokenize(sent)):
                    # Lower case and strip spaces
                    token = token.lower()
                    token = token.strip()
                    # If stopword, ignore token and continue
                    if token in self.stop_words:
                        continue
                    # If punctuation, continue
                    if all(char in self.punct for char in token):
                        continue
                    # Lemmatize/stem the token
                    if self.lemmatizer:
                        token = self.lemmatize(token, tag)
                    elif self.stemmer:
                        token = self.stem(token)
                    cleaned_tokens.append(token)
            # use CountVectorizer's _word_ngrams built in method to extract n-grams
            return self._word_ngrams(cleaned_tokens)

        return analyser
