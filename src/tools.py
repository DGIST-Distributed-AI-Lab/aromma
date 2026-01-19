import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def get_auroc_ap(y_true, y_pred):
    n_classes = y_true.shape[1]

    aucs, aps = [], []

    for i in list(range(n_classes)):
        if np.unique(y_true[:, i]).size < 2:
            continue
        else:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            ap = average_precision_score(y_true[:, i], y_pred[:, i])
        aucs.append(auc)
        aps.append(ap)
    return np.mean(aucs), np.mean(aps)

def filtered_score(y_true, y_pred, mask, label_indices):
    y_true_f = y_true[mask][:, label_indices]
    y_pred_f = y_pred[mask][:, label_indices]

    aucs, aps = [], []
    for i in range(len(label_indices)):
        col_true = y_true_f[:, i]
        col_pred = y_pred_f[:, i]
        if np.unique(col_true).size < 2:
            continue
        else:
            auc = roc_auc_score(col_true, col_pred)
            ap = average_precision_score(col_true, col_pred)
        aucs.append(auc)
        aps.append(ap)
    return np.mean(aucs), np.mean(aps)

def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}h {minutes:02d}m {secs:04d}s"