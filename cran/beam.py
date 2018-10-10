import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

from cran.cluster import spectral
from cran.mimo import MassiveMIMO, rayleigh


def cluster_usr():
    n_txs = 16
    n_rxs = 32
    n_clusters = 8

    # Spectral Clustering
    mimo = MassiveMIMO(n_txs=n_txs, n_rxs=n_rxs, seed=2)
    fading = rayleigh(mimo.pl_matrix_)
    affinity = usr_affinity(fading)
    labels = spectral(affinity_mat=affinity, n_clusters=n_clusters)
    mimo.draw(txs_labels=np.ones((n_txs,)) * 100, rxs_labels=labels, edges=False)

    print(labels)

    # TODO ensure non-empty clusters

    # TDMA + Zero-forcing Beamforming
    n_slot = 100
    rate = np.zeros((n_slot,))
    benchmark = np.zeros((n_slot,))
    rx_id_mat = one_hot(labels)
    for slot in range(n_slot):
        priorities = np.random.rand(n_rxs)
        # select the user with the highest priority in each cluster
        active_rx = np.argmax(priorities.reshape((n_rxs, 1)) * rx_id_mat, axis=0)
        rate[slot] = zero_forcing(fading[:, active_rx].T)

        # benchmark method
        active_rx = np.argsort(priorities)[:n_clusters]
        benchmark[slot] = zero_forcing(fading[:, active_rx].T)

    fig, ax = plt.subplots()
    plt.plot(np.arange(n_slot), rate, 'r-', label='Spectral Avg={0:.2f}'.format(np.mean(rate)))
    plt.plot(np.arange(n_slot), benchmark, 'b--', label='Random Avg={0:.2f}'.format(np.mean(benchmark)))
    ax.legend(loc='lower right', shadow=True, fontsize='x-large')
    plt.show()


def zero_forcing(h):
    gain = 1 / np.real(np.diag(inv(h.dot(h.conj().T))))
    rate = np.sum(np.log2(1 + gain))
    return rate


def one_hot(labels):
    n_clusters = np.max(labels) + 1
    identities = np.diag(np.ones((n_clusters,)))
    return identities[labels]


def usr_affinity(fading):
    HH = np.absolute(fading.conj().T.dot(fading))
    norm = 1 / np.sqrt(np.diag(HH))
    cos = np.diag(norm).dot(HH).dot(np.diag(norm))

    return np.abs(cos)


if __name__ == "__main__":
    cluster_usr()
