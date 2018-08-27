import time

import matplotlib.pyplot as plt
import numpy as np

from cluster import spectral, txs_centric, fast_update, spectral_co_clustering
from graph import to_bipartite
from mimo import DistributedMIMO, pl_approx_capacity, MassiveMIMO


def topology_spectral():
    n_txs = 128
    n_rxs = 100
    n_clusters = 10
    seed = 9

    mimo = DistributedMIMO(n_txs=n_txs, n_rxs=n_rxs, seed=seed)
    labels = spectral(to_bipartite(mimo.pl_matrix_), n_clusters=n_clusters)
    print('Spectral Rate = {0:.3f}'.format(
        pl_approx_capacity(mimo.pl_matrix_, txs_labels=labels[:n_txs], rxs_labels=labels[n_txs:])
    ))

    plt.close("all")
    plt.clf()
    mimo.draw(txs_labels=labels[:n_txs], rxs_labels=labels[n_txs:], edges=False)

    txs_labels, rxs_labels = spectral_co_clustering(mimo.pl_matrix_, n_clusters)
    print('Spectral Co Rate = {0:.3f}'.format(
        pl_approx_capacity(mimo.pl_matrix_, txs_labels=txs_labels, rxs_labels=rxs_labels)
    ))
    mimo.draw(txs_labels=txs_labels, rxs_labels=rxs_labels, edges=False)

    plt.show()


def topology_spectral_fast():
    n_txs = 100
    n_rxs = 100
    n_clusters = 10
    seed = 9

    # Initial
    mimo = DistributedMIMO(n_txs=n_txs, n_rxs=n_rxs, seed=seed)
    labels = spectral(to_bipartite(mimo.pl_matrix_), n_clusters=n_clusters)
    txs_labels = labels[:n_txs]
    rxs_labels = labels[n_txs:]
    print('Rate = {0:.3f}'.format(
        pl_approx_capacity(mimo.pl_matrix_, txs_labels=txs_labels, rxs_labels=rxs_labels)
    ))
    mimo.draw(txs_labels=txs_labels, rxs_labels=rxs_labels, edges=True)

    # Update
    std = 0.005
    mimo.move(std=std)
    rxs_labels = fast_update(mimo.pl_matrix_, txs_labels=txs_labels)
    print('Rate = {0:.3f}'.format(
        pl_approx_capacity(mimo.pl_matrix_, txs_labels=txs_labels, rxs_labels=rxs_labels)
    ))
    mimo.draw(txs_labels=txs_labels, rxs_labels=rxs_labels, edges=True)

    plt.show()


def topology_spectral_max_beam():
    n_txs = 16
    n_rxs = 16
    n_clusters = 8
    seed = 9

    mimo = MassiveMIMO(n_txs=n_txs, n_rxs=n_rxs, seed=seed)
    labels = spectral(to_bipartite(mimo.pl_matrix_), n_clusters=n_clusters)

    plt.close("all")
    plt.clf()
    mimo.draw(txs_labels=labels[:n_txs], rxs_labels=labels[n_txs:], edges=True)
    txs_labels = np.arange(n_txs)
    rxs_labels = txs_labels[np.argmax(mimo.pl_matrix_, axis=0)]
    mimo.draw(txs_labels=txs_labels, rxs_labels=rxs_labels, edges=True)
    plt.show()


def rate_spectral_static(n_txs, n_rxs, n_clusters, seed):
    mimo = DistributedMIMO(n_txs=n_txs, n_rxs=n_rxs, seed=seed)

    # Spectral clustering
    labels = spectral(to_bipartite(mimo.pl_matrix_), n_clusters=n_clusters)
    rate_sp = pl_approx_capacity(mimo.pl_matrix_, txs_labels=labels[:n_txs], rxs_labels=labels[n_txs:])

    # BS centric
    txs_labels, rxs_labels = txs_centric(mimo, n_clusters=n_clusters)
    rate_bs = pl_approx_capacity(mimo.pl_matrix_, txs_labels=txs_labels, rxs_labels=rxs_labels)

    return rate_sp, rate_bs


def rate_density():
    n_txs = 100
    n_clusters = 10
    x = np.arange(50, 151, 10)
    rate_sp = np.zeros(x.shape)
    rate_bs = np.zeros(x.shape)

    cnt = 0
    n_iter = 100
    start = time.time()
    for n_rxs in x:
        for i in range(n_iter):
            sp, bs = rate_spectral_static(n_txs=n_txs, n_rxs=n_rxs, n_clusters=n_clusters, seed=5)
            rate_sp[cnt] += sp
            rate_bs[cnt] += bs
        print('Time {0:.2f}, Spectral = {1:.3f}, BS = {2:.3f}'.format(
            time.time() - start, rate_sp[cnt], rate_bs[cnt]))
        cnt += 1
    rate_sp *= 1 / n_iter
    rate_bs *= 1 / n_iter

    fig, ax = plt.subplots()
    ax.plot(x / n_txs, rate_sp, 'b-', label='Dynamic Clustering')
    ax.plot(x / n_txs, rate_bs, 'k--', label='Static Clustering')
    ax.legend(loc='lower right', shadow=True, fontsize='x-large')
    plt.show()


def rate_fast():
    n_txs = 100
    n_rxs = 100
    n_clusters = 10
    seed = 9

    n_slots = 50
    rate_sp = np.zeros((n_slots,))
    rate_bs = np.zeros((n_slots,))

    # Initial
    mimo = DistributedMIMO(n_txs=n_txs, n_rxs=n_rxs, seed=seed)
    labels = spectral(to_bipartite(mimo.pl_matrix_), n_clusters=n_clusters)
    txs_labels_sp = labels[:n_txs]
    rxs_labels_sp = labels[n_txs:]
    rate_sp[0] = pl_approx_capacity(mimo.pl_matrix_, txs_labels=txs_labels_sp, rxs_labels=rxs_labels_sp)

    txs_labels_bs, rxs_labels_bs = txs_centric(mimo, n_clusters=n_clusters)
    rate_bs[0] = pl_approx_capacity(mimo.pl_matrix_, txs_labels=txs_labels_bs, rxs_labels=rxs_labels_bs)

    # Update
    std = 0.001
    for i in range(1, n_slots):
        mimo.move(std=std)
        rxs_labels_sp = fast_update(mimo.pl_matrix_, txs_labels=txs_labels_sp)
        rate_sp[i] = pl_approx_capacity(mimo.pl_matrix_, txs_labels=txs_labels_sp, rxs_labels=rxs_labels_sp)

        rxs_labels_bs = fast_update(mimo.pl_matrix_, txs_labels=txs_labels_bs)
        rate_bs[i] = pl_approx_capacity(mimo.pl_matrix_, txs_labels=txs_labels_bs, rxs_labels=rxs_labels_bs)
    plt.figure()
    plt.plot(np.arange(n_slots), rate_sp, 'b-', label='Dynamic Clustering')
    plt.plot(np.arange(n_slots), rate_bs, 'k--', label='Static Clustering')
    plt.legend(loc='best', shadow=True, fontsize='x-large')
    plt.show()


def complexity_slow_fast():
    x = np.arange(50, 501, 50)
    time_sp = np.zeros(x.shape)
    time_fast = np.zeros(x.shape)

    for i in range(x.shape[0]):
        n_clusters = int(x[i] / 10)
        mimo = DistributedMIMO(n_txs=x[i], n_rxs=x[i], seed=10)
        t0 = time.time()
        labels = spectral(to_bipartite(mimo.pl_matrix_), n_clusters=n_clusters)
        time_sp[i] = time.time() - t0

        # Update
        mimo.move(0.001)
        t0 = time.time()
        _ = fast_update(pl_matrix=mimo.pl_matrix_, txs_labels=labels[:x[i]])
        time_fast[i] = time.time() - t0
    plt.figure(0)
    plt.plot(x, time_sp, 'b-', label='Slow channel')
    plt.plot(x, time_fast, 'r-.', label='Fast channel')
    plt.yscale('log')
    plt.legend(loc='best', shadow=True, fontsize='x-large')

    plt.show()


def rate_spectral_beam():
    n_txs = 16
    n_rxs = 32
    seed = 5

    x = np.arange(2, n_txs + 1)
    rate_sp = np.zeros(x.shape)

    # spectral clustering
    mimo = MassiveMIMO(n_txs=n_txs, n_rxs=n_rxs, seed=seed)
    for (n_clusters, cnt) in zip(x, range(x.shape[0])):
        labels = spectral(to_bipartite(mimo.pl_matrix_), n_clusters=n_clusters)
        rate_sp[cnt] = pl_approx_capacity(mimo.pl_matrix_, txs_labels=labels[:n_txs], rxs_labels=labels[n_txs:])

    # max beam association
    txs_labels = np.arange(n_txs)
    rxs_labels = txs_labels[np.argmax(mimo.pl_matrix_, axis=0)]
    rate_beam = pl_approx_capacity(mimo.pl_matrix_, txs_labels=txs_labels, rxs_labels=rxs_labels)
    n_clusters_beam = count_clusters(txs_labels=txs_labels, rxs_labels=rxs_labels)
    print('target: {0:d}; actual : {1:d}'.format(n_txs, n_clusters_beam))

    # plot avg user rate versus n_clusters
    plt.figure()
    plt.plot(x, rate_sp, 'b-', label='Dynamic Clustering')
    plt.plot(n_clusters_beam, rate_beam, 'rx', markersize=10, label='Max Beam')
    plt.legend(loc='best', shadow=True, fontsize='x-large')

    # plot sum rate versus n_clusters
    plt.figure()
    plt.plot(x, rate_sp * x, 'b-', label='Dynamic Clustering')
    plt.plot(n_clusters_beam, rate_beam * n_clusters_beam, 'rx', markersize=10, label='Max Beam')
    plt.legend(loc='best', shadow=True, fontsize='x-large')
    plt.show()


def count_clusters(txs_labels, rxs_labels):
    n_clusters_max = np.int_(np.maximum(np.max(txs_labels), np.max(rxs_labels))) + 1
    n_clusters = 0
    for k in range(n_clusters_max):
        cluster_txs = txs_labels == k
        cluster_rxs = rxs_labels == k
        if np.sum(cluster_txs) <= 0 or np.sum(cluster_rxs) <= 0:
            continue
        n_clusters += 1
    return n_clusters


def rate_fast_beam():
    n_txs = 16
    n_rxs = 32
    n_clusters = 8
    seed = 9

    n_slots = 50
    rate_sp = np.zeros((n_slots,))
    rate_beam = np.zeros((n_slots,))

    # Initial spectral
    mimo = MassiveMIMO(n_txs=n_txs, n_rxs=n_rxs, seed=seed)
    labels = spectral(to_bipartite(mimo.pl_matrix_), n_clusters=n_clusters)
    txs_labels_sp = labels[:n_txs]
    rxs_labels_sp = labels[n_txs:]

    # max beam association
    txs_labels_beam = np.arange(n_txs)

    # Update
    std = 0.001
    for i in range(n_slots):
        mimo.move(std=std)

        # spectral
        rxs_labels_sp = fast_update(mimo.pl_matrix_, txs_labels=txs_labels_sp)
        rate_sp[i] = pl_approx_capacity(mimo.pl_matrix_, txs_labels=txs_labels_sp, rxs_labels=rxs_labels_sp)

        # max beam
        rxs_labels_beam = fast_update(mimo.pl_matrix_, txs_labels=txs_labels_beam)
        rate_beam[i] = pl_approx_capacity(mimo.pl_matrix_, txs_labels=txs_labels_beam, rxs_labels=rxs_labels_beam)
    plt.figure()
    plt.plot(np.arange(n_slots), rate_sp, 'b-', label='Dynamic Clustering')
    plt.plot(np.arange(n_slots), rate_beam, 'k--', label='Max Beam')
    plt.legend(loc='lower left', shadow=True, fontsize='x-large')
    plt.show()


if __name__ == "__main__":
    # topology_spectral_max_beam()
    # rate_fast_beam()
    topology_spectral()
