"""
classification
"""

from sklearn import metrics


def f1_score(y_true, y_pred):
    """Macro F1-score

    :param y_true: ground truth labels
    :param y_pred: predicted labels
    :return: macro f1-score
    """
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    return 2.0 * precision * recall / (precision + recall)
