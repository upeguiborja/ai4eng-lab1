import numpy as np
from typing import Any, cast
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_purity_score(df, label_col, n_clusters, random_state=42):
    X = df.drop(columns=[label_col])
    y = df[label_col].to_numpy()

    X_scaled = StandardScaler().fit_transform(X)
    labels = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=cast(Any, 10),
    ).fit_predict(X_scaled)

    purity_sum = 0
    for cluster in range(n_clusters):
        cluster_labels = y[labels == cluster].astype(int)
        if len(cluster_labels) > 0:
            purity_sum += np.max(np.bincount(cluster_labels))

    return round(float(purity_sum / len(y)), 4)
