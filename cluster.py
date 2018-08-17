import numpy as np
from numpy.linalg import eigh
from sklearn.cluster import KMeans

from graph import normalize_matrix


def spectral_clustering(affinity_mat, n_clusters):
    n_nodes = affinity_mat.shape[0]

    # Symmetric Laplacian
    laplacian_mat = np.eye(n_nodes) - normalize_matrix(affinity_mat)

    # Eigen decomposition values[i] * vectors[:,i] = A * vectors[:,i], values in ascending order
    values, vectors = eigh(laplacian_mat)

    # Eigen-space for clustering
    space = vectors[:, :n_clusters]
    row_norm = np.sqrt(np.sum(space * space, axis=1, keepdims=True))
    row_norm[row_norm == 0] = 1
    space = space / row_norm

    # k-means++
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=1).fit(space)

    return kmeans.labels_
