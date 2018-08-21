from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linear
import scipy.spatial.distance as dist
from sklearn.datasets.samples_generator import make_blobs

FIG_ID = 1

PATH_LOSS_EXPO = 3.6
PATH_LOSS_MIN = 0.010

class MIMO:
    def __init__(self, n_txs, n_rxs, seed=0):
        np.random.seed(int(seed))
        self.txs_ = np.random.rand(n_txs, 2)
        self.rxs_ = np.random.rand(n_rxs, 2)
        self.pl_matrix_ = path_loss(self.txs_, self.rxs_)

    def draw(self, txs_labels, rxs_labels, edges=False):
        global FIG_ID
        plt.figure(FIG_ID)
        FIG_ID += 1
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        n_clusters = np.int_(np.maximum(np.max(txs_labels), np.max(rxs_labels))) + 1
        for k, color in zip(range(n_clusters), colors):
            cluster_txs = txs_labels == k
            cluster_rxs = rxs_labels == k
            plt.plot(self.txs_[cluster_txs, 0], self.txs_[cluster_txs, 1], color + '^', markersize=5,
                     markerfacecolor=color)
            plt.plot(self.rxs_[cluster_rxs, 0], self.rxs_[cluster_rxs, 1], color + 'o', markersize=5,
                     markerfacecolor=color)
            if not edges:
                continue
            for tx in self.txs_[cluster_txs]:
                for rx in self.rxs_[cluster_rxs]:
                    plt.plot([tx[0], rx[0]], [tx[1], rx[1]], color, linewidth=1)
        plt.title('Number of clusters: %d' % n_clusters)

    def move(self, std):
        self.rxs_ += np.random.randn(self.rxs_.shape[0], 2) * std
        self.pl_matrix_ = path_loss(self.txs_, self.rxs_)


class MassiveMIMO(MIMO):
    def __init__(self, n_txs, n_rxs, seed=0):
        np.random.seed(int(seed))
        # BS at [.5, .5]
        self.rxs_ = np.random.rand(n_rxs, 2) - .5
        self.pl_matrix_ = path_loss(usrs_theta, n_beams);


class ClusteredMIMO(MIMO):
    def __init__(self, n_txs, n_rxs, n_clusters, std=.5, seed=0):
        centers = np.random.rand(n_clusters, 2)
        self.txs_, labels_true = make_blobs(
            n_samples=n_txs, centers=centers.tolist(), cluster_std=std, random_state=seed
        )
        self.rxs_, labels_true = make_blobs(
            n_samples=n_rxs, centers=centers.tolist(), cluster_std=std, random_state=seed + 1
        )
        self.pl_matrix_ = path_loss(self.txs_, self.rxs_)


def path_loss(txs, rxs):
    global PATH_LOSS_EXPO, PATH_LOSS_MIN
    distance_mat = dist.cdist(txs, rxs)
    pl = 1 / np.power(np.maximum(distance_mat, PATH_LOSS_EXPO), PATH_LOSS_EXPO)
    return pl


def path_loss_angular(n_txs, rxs):
    global PATH_LOSS_EXPO, PATH_LOSS_MIN
    distance_vec = np.sqrt(rxs[:, 0] * rxs[:, 0] + rxs[:, 1] * rxs[:, 1])

    angular_vec = np.arctan2(rxs[:, 1], rxs[:, 0]) + np.pi
    angular_vec = .5 * n_txs * np.pi * np.cos(angular_vec)
    beta = (np.arange(1, n_txs + 1) - (n_txs + 1) / 2) * np.pi
    delta = np.tile(beta, (rxs.shape[0], 1)).T * np.tile(beta, (n_txs, 1))
    A = n_txs * np.power(np.sin(delta) / np.sin(delta / n_txs) / n_txs, 2)


def slow_fading_capacity(channel_mat, txs_labels, rxs_labels, snr=1e10):
    n_clusters = np.int_(np.maximum(np.max(txs_labels), np.max(rxs_labels))) + 1
    rate = np.zeros((n_clusters,))
    for k in range(n_clusters):
        cluster_txs = txs_labels == k
        cluster_rxs = rxs_labels == k
        signal_mat = channel_mat[cluster_txs][:, cluster_rxs]

        # Downlink
        intf_vec = np.sum(channel_mat[~cluster_txs][:, cluster_rxs], axis=0, keepdims=False)
        intf_vec += 1 / snr
        channel_eff = np.matmul(signal_mat, np.diag(1 / np.power(intf_vec, 1 / 2)))
        rate[k] = shannon_capacity(channel_eff) / np.sum(cluster_rxs)

        # Uplink
        intf_vec = np.sum(channel_mat[cluster_txs][:, ~cluster_rxs], axis=1, keepdims=False)
        intf_vec += 1 / snr
        channel_eff = np.matmul(np.diag(1 / np.power(intf_vec, 1 / 2)), signal_mat)
        rate[k] += shannon_capacity(channel_eff) / np.sum(cluster_txs)
    return np.sum(utility(rate)) / n_clusters


def fast_fading_capacity(pl_mat, txs_labels, rxs_labels, snr=1e10, n_fadings=int(1e3)):
    rate = np.zeros((n_fadings,))
    n_rows = pl_mat.shape[0]
    n_cols = pl_mat.shape[1]
    pl_mat_sqrt = np.sqrt(pl_mat)
    rayleigh_pool = np.random.randn(n_fadings, n_rows, n_cols) + 1j * np.random.randn(n_fadings, n_rows, n_cols)
    rayleigh_pool *= 1 / np.sqrt(2)
    for i in range(n_fadings):
        channel_mat = rayleigh_pool[i] * pl_mat_sqrt
        rate[i] = slow_fading_capacity(channel_mat=channel_mat, txs_labels=txs_labels, rxs_labels=rxs_labels, snr=snr)
    return np.sum(utility(rate)) / n_fadings


def pl_approx_capacity(pl_mat, txs_labels, rxs_labels, snr=1e10):
    n_clusters = np.int_(np.maximum(np.max(txs_labels), np.max(rxs_labels))) + 1
    rate = np.zeros((n_clusters, 2))
    for k in range(n_clusters):
        cluster_txs = txs_labels == k
        cluster_rxs = rxs_labels == k
        signal_mat = pl_mat[cluster_txs][:, cluster_rxs]

        # Downlink
        intf_vec = np.sum(pl_mat[~cluster_txs][:, cluster_rxs], axis=0, keepdims=False)
        intf_vec += 1 / snr
        sinr = np.sum(signal_mat, axis=0) / intf_vec
        rate[k, 0] = np.sum(np.log2(1 + sinr)) / sinr.shape[0]

        # Uplink
        intf_vec = np.sum(pl_mat[cluster_txs][:, ~cluster_rxs], axis=1, keepdims=False)
        intf_vec += 1 / snr
        sinr = np.sum(signal_mat, axis=1) / intf_vec
        rate[k, 1] = np.sum(np.log2(1 + sinr)) / sinr.shape[0]
    return np.sum(utility(rate)) / n_clusters


def shannon_capacity(channel_mat):
    eigenvalues = linear.eigvalsh(np.matmul(channel_mat, channel_mat.conj().T))
    return np.sum(np.log2(1 + eigenvalues))


def utility(rate):
    return rate


if __name__ == '__main__':
    # mimo = MIMO(n_txs=100, n_rxs=100, seed=1)
    # mimo.draw(np.ones((100,)), np.ones((100,)), edges=False)

    mimo2 = ClusteredMIMO(n_txs=100, n_rxs=100, seed=1)
    mimo2.draw(np.ones((100,)), np.ones((100,)), edges=False)
