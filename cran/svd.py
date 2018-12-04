# Singular vector decomposition algorithms
import random

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl

from cran.mimo import DistributedMIMO


class Node:
    def __init__(self, index, adj):
        self.index = index
        self.adj = adj.reshape((1, -1))
        self.neighbors = None
        self.rank = None
        self.x = None
        self.Y = None
        self.step_size = 0

    def init_svd(self, rank, x, Y, step_size):
        self.rank = rank
        self.x = x.reshape((1, -1))
        self.Y = Y
        self.step_size = step_size

    def send(self):
        receiver = random.sample(self.neighbors, 1)[0]
        receiver.receive(self.Y)

    def receive(self, Y):
        self.Y = Y
        pair = update_rank(self.Y, self.x, self.adj, self.step_size)
        self.x = pair['x']
        self.Y = pair['Y']


def update_rank(Y, x, a, step_size=0.1):
    err = a - x.dot(Y.T)
    x_new = x + step_size * err.dot(Y)
    Y_new = Y + step_size * err.T.dot(x)
    return {'x': x_new, 'Y': Y_new}


def update_svd(Y, x, a, step_size=0.01):
    a_new = a + 0
    x_new = np.zeros(x.shape)
    Y_new = np.zeros(Y.shape)
    for dim in range(Y.shape[1]):
        err = a_new - x[0, dim] * Y[:, dim].reshape((1, -1))
        x_new[0, dim] = x[0, dim] + step_size * err.dot(Y[:, dim])
        Y_new[:, dim] = Y[:, dim] + step_size * x[0, dim] * err.reshape((-1,))
        a_new = a_new - x[0, dim] * Y[:, dim].reshape((1, -1))
    return {'x': x_new, 'Y': Y_new}


def gradient_low_rank(A, rank, step_size=0.01, n_iter=1000):
    rank = min(rank, min(A.shape))
    X = np.random.rand(A.shape[0], rank)
    Y = np.random.rand(A.shape[1], rank)

    results = np.zeros((n_iter,))
    error_bnd = npl.norm(A - X.dot(Y.T)) * 0.01

    for cnt in range(n_iter):
        results[cnt] = npl.norm(A - X.dot(Y.T))
        if results[cnt] <= error_bnd:
            break
        delta = A - X.dot(Y.T)
        X_new = X + step_size * delta.dot(Y)
        Y_new = Y + step_size * delta.T.dot(X)
        X = X_new
        Y = Y_new
    plt.figure()
    plt.plot(np.arange(0, n_iter), results)
    plt.xlabel("Index of Iterations")
    plt.ylabel("Error norm")
    plt.title("Gradient descent: matrix dim = {0} x {1}, rank = {2}".format(A.shape[0], A.shape[1], rank))
    plt.legend(["Init={0:.2f}, final={1:.2f}".format(results[0], results[-1])])


def gradient_svd(A, rank, step_size=0.01, n_iter=10000):
    rank = min(min(rank, A.shape[0]), A.shape[1])
    X = np.random.rand(A.shape[0], rank)
    Y = np.random.rand(A.shape[1], rank)

    results = np.zeros((n_iter, rank))
    u, s, vh = npl.svd(A, full_matrices=False, compute_uv=True)

    for cnt in range(n_iter):
        error_curr = npl.norm(A - X.dot(Y.T))
        A_new = A
        for dim in range(0, rank):
            x_dim = X[:, dim].reshape(-1, 1)
            y_dim = Y[:, dim].reshape(-1, 1)
            delta = A_new - x_dim.dot(y_dim.T)
            X[:, dim] += (step_size * delta.dot(y_dim)).reshape((-1,))
            Y[:, dim] += (step_size * delta.T.dot(x_dim)).reshape((-1,))
            A_new = delta
            results[cnt, dim] = abs(vh.T[:, dim].dot(Y[:, dim]) / npl.norm(Y[:, dim]))
    plt.figure(0)
    plt.plot(np.arange(0, n_iter), results)
    plt.xlabel("Index of Iterations")
    plt.ylabel("Cosine(y, v)")
    plt.title("Gradient descent SVD: matrix dim = {0} x {1}, rank = {2}".format(A.shape[0], A.shape[1], rank))
    plt.legend(["Rank %d" % k for k in range(rank)])


def distributed_svd(A, rank, step_size=0.01, n_iter=1000):
    rank = min(rank, min(A.shape))
    n_nodes = A.shape[0]
    X = np.random.rand(n_nodes, rank)
    Y = np.random.rand(A.shape[1], rank)

    # build nodes
    nodes = []
    for index in range(n_nodes):
        nodes.append(Node(index=index, adj=A[index]))
    for node in nodes:
        neighbors = set(nodes)
        neighbors.remove(node)
        node.neighbors = neighbors
        node.init_svd(rank=rank, x=X[index], Y=Y, step_size=step_size)

    u, s, vh = npl.svd(A, full_matrices=False, compute_uv=True)
    results = np.zeros((n_iter, rank))
    error = np.zeros((n_iter,))
    for cnt in range(n_iter):
        for node in nodes:
            node.send()
            X[index] = node.x.reshape((-1,))
        Y = nodes[-1].Y
        error[cnt] = npl.norm(A - X.dot(Y.T))
        for dim in range(rank):
            results[cnt, dim] = 1 - abs(u[:, dim].dot(Y[:, dim]) / npl.norm(Y[:, dim]))

    plt.figure(0)
    plt.plot(np.arange(0, n_iter), results)
    plt.xlabel("Iterations")
    plt.ylabel("1 - Cosine(y, v)")
    plt.title("Distributed SVD: matrix dim = {0} x {1}, rank = {2}".format(A.shape[0], A.shape[1], rank))
    plt.legend(["Rank %d: %.2f" % (k, results[-1, k]) for k in range(rank)])

    plt.figure(1)
    plt.plot(np.arange(0, n_iter), error)

    return Y


def centralized_svd(matrix, rank):
    u, s, vh = npl.svd(matrix, full_matrices=False, compute_uv=True)
    return u[..., :rank]


def main():
    n_clusters = 3
    n_txs = 10
    n_rxs = 10
    mimo = DistributedMIMO(n_txs, n_rxs, seed=0)

    # Calculate symmetric laplacian of the bipartite graph
    adj = np.vstack([np.hstack([np.zeros((n_txs, n_txs)), mimo.pl_matrix_]),
                     np.hstack([mimo.pl_matrix_.T, np.zeros((n_rxs, n_rxs))])
                     ])
    degree = np.sum(adj, axis=1)
    if np.any(degree <= 0):
        raise Exception('exist unconnected nodes')
    degree_sqrt_inverse = np.diag(np.power(degree, - 1 / 2))
    laplacian = np.eye(n_txs + n_rxs) + degree_sqrt_inverse.dot(adj).dot(degree_sqrt_inverse)

    # gradient_low_rank(A, rank, rate=0.01)
    # gradient_svd(A, rank, rate=0.01)
    distributed_svd(laplacian, rank=n_clusters, n_iter=100)
    plt.show()


def test_matrix():
    n_dim = 100
    rank = 5
    a = np.random.rand(n_dim, rank)
    matrix = a.dot(a.T)
    # distributed_svd(matrix, rank)
    # gradient_low_rank(matrix, rank, step_size=0.001, n_iter=1000)
    # gradient_svd(matrix, rank, step_size=0.001, n_iter=10000)
    distributed_svd(matrix, rank, step_size=0.01, n_iter=1000)
    plt.show()


if __name__ == "__main__":
    np.random.seed(0)
    test_matrix()
