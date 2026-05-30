from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


def agrupar_jerarquicamente(df, n_clusters, metodo):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    X_scaled = StandardScaler().fit_transform(df[numeric_cols])

    labels = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=metodo,
    ).fit_predict(X_scaled)

    df_con_cluster = df.copy()
    df_con_cluster["cluster"] = labels
    perfil_clusters = df_con_cluster.groupby("cluster")[numeric_cols].mean()

    return df_con_cluster, perfil_clusters
