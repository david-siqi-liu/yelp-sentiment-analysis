"""
visualize
"""

import matplotlib.pyplot as plt
from seaborn import heatmap
from sklearn.metrics import confusion_matrix


def confusion_heat_map(y_true, y_pred, normalize, fmt, labels):
    """Generate heat map based on a confusion matrix
    """
    conf_mat = confusion_matrix(y_true,
                                y_pred,
                                normalize=normalize)

    plt.figure(figsize=(3, 3))

    heatmap(conf_mat,
            annot=True,
            annot_kws={"size": 12},
            fmt=fmt,
            xticklabels=labels,
            yticklabels=labels,
            cbar=False)

    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.show()
