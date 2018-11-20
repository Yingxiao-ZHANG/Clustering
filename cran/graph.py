import numpy as np


def to_bipartite(weight_mat):
    n_rows = weight_mat.shape[0]
    n_cols = weight_mat.shape[1]
    affinity_mat = np.vstack([np.hstack([np.zeros((n_rows, n_rows)), weight_mat]),
                              np.hstack([weight_mat.T, np.zeros((n_cols, n_cols))])
                              ])
    return affinity_mat


def normalize_matrix(A):
    row_sum = np.sum(A, axis=1)
    drow_inv_sqrt = np.power(row_sum, -0.5)
    drow_inv_sqrt[np.isinf(drow_inv_sqrt)] = 0.
    drow_inv_sqrt_mat = np.diag(drow_inv_sqrt)

    col_sum = np.sum(A, axis=0)
    dcol_inv_sqrt = np.power(col_sum, -0.5)
    dcol_inv_sqrt[np.isinf(dcol_inv_sqrt)] = 0.
    dcol_inv_sqrt_mat = np.diag(dcol_inv_sqrt)

    return drow_inv_sqrt_mat.dot(A).dot(dcol_inv_sqrt_mat)


def cut_metrics(affinity, labels):
    if affinity.shape[0] != affinity.shape[1] or labels.shape[0] != affinity.shape[0]:
        return {}
    n_clusters = np.int_(np.max(labels)) + 1
    normalized_cut = 0
    cut_ratio = 0
    for k in range(n_clusters):
        if not np.any(labels == k):
            continue
        active = affinity[labels == k]
        vol = np.sum(active)
        cut = np.sum(active[:, labels != k])

        normalized_cut += cut / vol

        if vol == cut:
            continue
        if vol < cut or vol - cut < 1e-10:
            continue
            print('vol < cut or vol - cut < 1e-10')
            print('Cluster ID {0:d}, Vol={1:.2f}, Cut={2:.2f}'.format(k, vol, cut))
        if vol > cut:
            cut_ratio += cut / (vol - cut)

    return {'normalized cut': normalized_cut, 'cut ratio': cut_ratio}


if __name__ == '__main__':
    A = to_bipartite(np.random.rand(10, 8))
    print(A.shape)
