import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def purity(labels_true, labels_pred):
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    total = 0
    for c in np.unique(labels_pred):
        counts = np.bincount(labels_true[labels_pred == c])
        total += counts.max()
    return float(total / len(labels_true))


def external_scores(labels_true, labels_pred):
    return {
        "ari": float(adjusted_rand_score(labels_true, labels_pred)),
        "nmi": float(normalized_mutual_info_score(labels_true, labels_pred)),
        "purity": purity(labels_true, labels_pred),
    }


def stability(label_sets):
    """Mean pairwise ARI between clusterings from different seeds."""
    vals = []
    for i in range(len(label_sets)):
        for j in range(i + 1, len(label_sets)):
            vals.append(adjusted_rand_score(label_sets[i], label_sets[j]))
    return float(np.mean(vals)) if vals else float("nan")
