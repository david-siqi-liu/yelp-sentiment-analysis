"""
train_model
"""
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from yelpsent import visualization


def train_and_test(X_train, y_train, X_test, y_test, vectorizer, classifier):
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])

    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    print("Classification Report - Training")
    print(classification_report(y_train, y_train_pred, digits=4))

    y_test_pred = pipeline.predict(X_test)
    print("Classification Report - Testing")
    print(classification_report(y_test, y_test_pred, digits=4))

    print("Confusion Matrix - Testing")
    visualization.confusion_heat_map(y_test,
                                     y_test_pred,
                                     normalize='true',
                                     fmt='.1%',
                                     labels=set(y_test))

    return pipeline
