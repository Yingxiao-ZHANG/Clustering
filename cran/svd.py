# Singular vector decomposition algorithms
import queue
import random

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl

from cran.graph import normalize_matrix, to_bipartite
from cran.mimo import DistributedMIMO


class Node:
    def __init__(self, index, adj, rank):
        self.index = index
        self.rank = rank
        self.adj = adj.reshape((1, -1))
        self.x = np.random.rand(1, rank)
        self.Y = np.empty((adj.shape[0], rank))
        self.queue = queue.Queue(maxsize=adj.shape[0])

    def update(self, step_size):
        if self.queue.empty():
            return self.Y
        while not self.queue.empty():
            Y = self.queue.get(False)
            err = self.adj - self.x.dot(Y.T)
            self.x += step_size * err.dot(Y)
        Y += step_size * err.T.dot(self.x)
        self.Y = Y
        return self.Y

    def update_svd(self, step_size):
        if self.queue.empty():
            return self.Y
        while not self.queue.empty():
            Y = self.queue.get(False)
            a = self.adj
            x = self.x
            for dim in range(self.rank):
                err = a - x[0, dim] * Y[:, dim].reshape((1, -1))
                x[0, dim] += step_size * err.dot(Y[:, dim])
                Y[:, dim] += step_size * (err * x[0, dim]).reshape((-1,))
                a = err
            self.Y = Y
            self.x = x
        return self.Y

    def receive(self, Y):
        self.queue.put(Y)


def svd_QR(matrix):
    u, s, vh = npl.svd(matrix, full_matrices=False, compute_uv=True)
    return {"u": u, "lambda": s, "v": vh.T}


def gradient_low_rank(A, rank, rate=0.01, n_iter=10000):
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
        X_new = X + rate * delta.dot(Y)
        Y_new = Y + rate * delta.T.dot(X)
        X = X_new
        Y = Y_new
    plt.figure()
    plt.plot(np.arange(0, n_iter), results)
    plt.xlabel("Index of Iterations")
    plt.ylabel("Error norm")
    plt.title("Gradient descent: matrix dim = {0} x {1}, rank = {2}".format(A.shape[0], A.shape[1], rank))
    plt.legend(["Init={0:.2f}, final={1:.2f}".format(results[0], results[-1])])


def distributed_low_rank(A, rank, step_size=0.01, n_iter=10000):
    rank = min(rank, min(A.shape))
    n_nodes = A.shape[0]
    X = np.random.rand(n_nodes, rank)
    Y = np.random.rand(A.shape[1], rank)
    nodes = [Node(index, A[index], rank) for index in range(n_nodes)]

    results = np.zeros((n_iter,))
    for node in nodes:
        node.receive(Y)
    for cnt in range(n_iter):
        for index, node in enumerate(nodes):
            Y = node.update(step_size)
            neighbor = random.randint(0, n_nodes - 1)
            if neighbor == node.index:
                neighbor = (neighbor + 1) % n_nodes
            nodes[neighbor].receive(Y)
            X[index] = node.x.reshape((-1,))
        results[cnt] = npl.norm(A - X.dot(Y.T))

    plt.figure()
    plt.plot(np.arange(0, n_iter), results)
    plt.xlabel("Index of Iterations")
    plt.ylabel("Error norm")
    plt.title("Gradient descent: matrix dim = {0} x {1}, rank = {2}".format(A.shape[0], A.shape[1], rank))
    plt.legend(["Init={0:.2f}, final={1:.2f}".format(results[0], results[-1])])


def gradient_svd(A, rank, rate=0.01, n_iter=10000):
    rank = min(min(rank, A.shape[0]), A.shape[1])
    X = np.random.rand(A.shape[0], rank)
    Y = np.random.rand(A.shape[1], rank)

    results = np.zeros((n_iter, rank))
    error_bnd = npl.norm(A - X.dot(Y.T)) * 0.01
    u, s, vh = npl.svd(A, full_matrices=False, compute_uv=True)

    for cnt in range(n_iter):
        error_curr = npl.norm(A - X.dot(Y.T))
        if error_curr <= error_bnd:
            break
        A_new = A
        for dim in range(0, rank):
            x_dim = X[:, dim].reshape(-1, 1)
            y_dim = Y[:, dim].reshape(-1, 1)
            delta = A_new - x_dim.dot(y_dim.T)
            X[:, dim] += (rate * delta.dot(y_dim)).reshape((-1,))
            Y[:, dim] += (rate * delta.T.dot(x_dim)).reshape((-1,))
            A_new = delta
            results[cnt, dim] = abs(vh.T[:, dim].dot(Y[:, dim]) / npl.norm(Y[:, dim]))
    plt.figure()
    plt.plot(np.arange(0, n_iter), results)
    plt.xlabel("Index of Iterations")
    plt.ylabel("Cosine(y, v)")
    plt.title("Gradient descent SVD: matrix dim = {0} x {1}, rank = {2}".format(A.shape[0], A.shape[1], rank))
    plt.legend(["Rank %d" % k for k in range(rank)])


def distributed_svd(A, rank, step_size=0.01, n_iter=10000):
    rank = min(rank, min(A.shape))
    n_nodes = A.shape[0]
    X = np.random.rand(n_nodes, rank)
    Y = np.random.rand(A.shape[1], rank)
    nodes = [Node(index, A[index], rank) for index in range(n_nodes)]
    u, s, vh = npl.svd(A, full_matrices=False, compute_uv=True)

    results = np.zeros((n_iter, rank))
    for node in nodes:
        node.receive(Y)
    for cnt in range(n_iter):
        for index, node in enumerate(nodes):
            Y = node.update_svd(step_size)
            neighbor = random.randint(0, n_nodes - 1)
            if neighbor == node.index:
                neighbor = (neighbor + 1) % n_nodes
            nodes[neighbor].receive(Y)
            X[index] = node.x.reshape((-1,))
        for dim in range(rank):
            results[cnt, dim] = abs(vh.T[:, dim].dot(Y[:, dim]) / npl.norm(Y[:, dim]))

    plt.figure()
    plt.plot(np.arange(0, n_iter), results)
    plt.xlabel("Index of Iterations")
    plt.ylabel("Cosine(y, v)")
    plt.title("Distributed SVD: matrix dim = {0} x {1}, rank = {2}".format(A.shape[0], A.shape[1], rank))
    plt.legend(["Rank %d: %.2f" % (k, results[-1, k]) for k in range(rank)])


if __name__ == "__main__":
    rank = 3
    n_txs = 10
    n_rxs = 10
    mimo = DistributedMIMO(n_txs, n_rxs, seed=0)
    laplacian = np.eye(n_txs + n_rxs) + normalize_matrix(to_bipartite(mimo.pl_matrix_))

    # gradient_low_rank(A, rank, rate=0.01)
    # gradient_svd(A, rank, rate=0.01)
    distributed_svd(laplacian, rank, step_size=0.005, n_iter=10000)
    plt.show()
