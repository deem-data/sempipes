import numpy as np
from sklearn.metrics import roc_auc_score


def scoring(y_test, y_pred):
    if len(np.unique(y_test)) == 2:
        # Binary classification
        return roc_auc_score(y_test, y_pred[:, 1])
    else:
        # Multiclass classification
        return roc_auc_score(y_test, y_pred, multi_class="ovo", average="macro")


__all__ = ["scoring"]
