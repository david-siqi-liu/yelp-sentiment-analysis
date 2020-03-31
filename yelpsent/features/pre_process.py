"""
pre_process
"""

from nltk import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(CountVectorizer, self).build_analyzer()
        return lambda doc: (stemmer.stem(w) for w in analyzer(doc))


def get_wordnet_pos(w):
    tag = pos_tag([w])[0][1][0].upper()
    return {'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV,
            'J': wordnet.ADJ
            }.get(tag, wordnet.NOUN)


class LemmatizedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(CountVectorizer, self).build_analyzer()
        return lambda doc: (lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in analyzer(doc))


def get_examples(phrase, dtm, cv, n=5) -> list:
    idx = cv.get_feature_names().index(phrase)
    examples = []

    for i in range(dtm.shape[0]):
        if dtm[i].toarray()[0][idx] == 1:
            examples.append(i)
            if len(examples) > n:
                break

    return examples
