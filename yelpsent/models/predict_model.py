"""
predict_model
"""

import math

import numpy as np


def get_examples(X, y_actual, y_pred_1, y_pred_2, val_actual, val_pred_1, val_pred_2, k=5):
    n = len(X)
    assert len(y_actual) == n and len(y_pred_1) == n and len(y_pred_2) == n
    examples = {}
    for x, y, y_1, y_2 in zip(X, y_actual, y_pred_1, y_pred_2):
        if y == val_actual and y_1 == val_pred_1 and y_2 == val_pred_2:
            examples[x] = len(x)
    return sorted(examples.items(), key=lambda x: x[1])[:k]


def nb_predict_proba(vec, clf, doc):
    features = vec.get_feature_names()
    log_probs = clf.feature_log_prob_.transpose()
    num_classes = len(log_probs[0])
    # Priors
    log_priors = clf.class_log_prior_
    print("Log Priors: {0}".format(log_priors))
    priors = [math.exp(i) for i in log_priors]
    # Log Likelihoods
    doc_transformed = vec.transform(doc)
    doc_features, doc_log_probs = [], []
    for i, v in enumerate(doc_transformed.toarray()[0]):
        if v > 0:
            doc_features.append(features[i])
            doc_log_probs.append(log_probs[i])
            print("Feature: {0}, Log Likelihoods: {1}".format(features[i], log_probs[i]))
    # Joint Likelihoods and Evidence
    joint_probs = []
    evidence = 0
    for c, l in enumerate(np.array(doc_log_probs).transpose()):
        joint_prob = np.prod(np.exp(l))
        joint_probs.append(joint_prob)
        evidence += joint_prob * priors[c]
    # Posteriors
    posteriors = []
    for c in range(num_classes):
        posterior = joint_probs[c] * priors[c] / evidence
        posteriors.append(posterior)
    return posteriors


def most_frequent_words(vec, clf, k=10):
    features = vec.get_feature_names()
    log_probs = clf.feature_log_prob_.transpose()
    features_to_log_probs = {}
    for f, l in zip(features, log_probs):
        features_to_log_probs[f] = l
    for i in sorted(features_to_log_probs.items(),
                    key=lambda x: np.average(x[1]),
                    reverse=True)[:k]:
        print(i)
