import sklearn.cluster as sk

from cluster import spectral_clustering
from graph import to_bipartite
from mimo import ClusteredMIMO


def test_spectral_clustering():
    n_txs = 30
    n_rxs = 30
    mimo = ClusteredMIMO(n_txs=n_txs, n_rxs=n_rxs, seed=4)
    # mimo2.draw(np.ones((100,)), np.ones((100,)), edges=False)

    # Spectral clustering
    labels = spectral_clustering(to_bipartite(mimo.pl_matrix_), n_clusters=3)
    mimo.draw(labels[:n_txs], labels[n_txs:])

    labels = sk.spectral_clustering(to_bipartite(mimo.pl_matrix_), n_clusters=3)
    mimo.draw(labels[:n_txs], labels[n_txs:])


if __name__ == "__main__":
    test_spectral_clustering()
