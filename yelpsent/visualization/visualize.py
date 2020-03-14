"""
visualize
"""

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from seaborn import heatmap


def confusion_heat_map(y_true, y_pred, normalize, fmt, labels):
    """Generate heat map based on a confusion matrix
    """
    conf_mat = confusion_matrix(y_true,
                                y_pred,
                                normalize=normalize)
    heatmap(conf_mat,
            annot=True,
            fmt=fmt,
            xticklabels=labels,
            yticklabels=labels)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
