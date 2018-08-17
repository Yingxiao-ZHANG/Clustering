from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as dist
from sklearn.datasets.samples_generator import make_blobs


class MIMO:
    def __init__(self, n_txs, n_rxs, seed=0):
        np.random.seed(int(seed))
        self.txs_ = np.random.rand(n_txs, 2)
        self.rxs_ = np.random.rand(n_rxs, 2)
        self.pl_matrix_ = path_loss_matrix(self.txs_, self.rxs_)

    def draw(self, txs_labels, rxs_labels, edges=False):
        plt.close('all')
        plt.figure(1)
        plt.clf()
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        n_clusters = np.int_(np.maximum(np.max(txs_labels), np.max(rxs_labels))) + 1
        for k, color in zip(range(n_clusters), colors):
            cluster_txs = txs_labels == k
            cluster_rxs = rxs_labels == k
            plt.plot(self.txs_[cluster_txs, 0], self.txs_[cluster_txs, 1], color + '^', markersize=5,
                     markerfacecolor=color)
            plt.plot(self.rxs_[cluster_rxs, 0], self.rxs_[cluster_rxs, 1], color + 'o', markersize=5,
                     markerfacecolor=color)
            if ~edges:
                continue
            for tx in self.txs_[cluster_txs]:
                for rx in self.rxs_[cluster_rxs]:
                    plt.plot([tx[0], rx[0]], [tx[1], rx[1]], color, linewidth=1)
        plt.title('Number of clusters: %d' % n_clusters)
        plt.show()


def path_loss_matrix(txs, rxs):
    distance_matrix = dist.cdist(txs, rxs)
    pl = 1 / np.power(np.maximum(distance_matrix, .010), 3.6)
    return pl


class ClusteredMIMO(MIMO):
    def __init__(self, n_txs, n_rxs, seed=0):
        centers = [[1, 1], [-1, -1], [1, -1]]
        self.txs_, labels_true = make_blobs(n_samples=n_txs, centers=centers, cluster_std=0.5, random_state=seed)
        self.rxs_, labels_true = make_blobs(n_samples=n_rxs, centers=centers, cluster_std=0.5, random_state=seed + 1)
        self.pl_matrix_ = path_loss_matrix(self.txs_, self.rxs_)


if __name__ == '__main__':
    # mimo = MIMO(n_txs=100, n_rxs=100, seed=1)
    # mimo.draw(np.ones((100,)), np.ones((100,)), edges=False)

    mimo2 = ClusteredMIMO(n_txs=100, n_rxs=100, seed=1)
    mimo2.draw(np.ones((100,)), np.ones((100,)), edges=False)
