"""
train_model
"""
from sklearn.pipeline import Pipeline

from yelpsent import metrics
from yelpsent import visualization


def evaluate_pipeline(X_train, y_train, X_test, y_test, pipeline):
    y_train_pred = pipeline.predict(X_train)
    f1_train = metrics.f1_score(y_train, y_train_pred)

    y_test_pred = pipeline.predict(X_test)
    f1_test = metrics.f1_score(y_test, y_test_pred)

    visualization.confusion_heat_map(y_test,
                                     y_test_pred,
                                     normalize='true',
                                     fmt='.1%',
                                     labels=set(y_test))

    return y_train_pred, y_test_pred, f1_train, f1_test


def train_and_test(X_train, y_train, X_test, y_test, classifier, vectorizer=None):
    if vectorizer:
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
    else:
        pipeline = Pipeline([
            ('classifier', classifier)
        ])

    pipeline.fit(X_train, y_train)

    y_train_pred, y_test_pred, f1_train, f1_test = evaluate_pipeline(X_train, y_train, X_test, y_test, pipeline)

    return pipeline, y_train_pred, y_test_pred, f1_train, f1_test
