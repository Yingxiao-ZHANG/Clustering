import matplotlib.pyplot as plt
import numpy as np

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
    plt.show()

    # Rate calculation


def usr_affinity(fading):
    HH = np.absolute(fading.conj().T.dot(fading))
    norm = 1 / np.sqrt(np.diag(HH))
    cos = np.diag(norm).dot(HH).dot(np.diag(norm))

    return np.abs(cos)


if __name__ == "__main__":
    cluster_usr()
