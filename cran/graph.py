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


if __name__ == '__main__':
    A = to_bipartite(np.random.rand(10, 8))
    print(A.shape)
