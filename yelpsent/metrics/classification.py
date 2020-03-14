"""
classification
"""

from sklearn import metrics


def accuracy_score(y_true, y_pred):
    """Accuracy classification score, normalized

    :param y_true: ground truth labels
    :param y_pred: predicted labels
    :return: fraction of correctly classified samples
    """
    return metrics.accuracy_score(y_true,
                                  y_pred,
                                  normalize=True)
