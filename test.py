import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AffinityPropagation

from cluster import spectral, txs_centric, fast_update, spectral_co_clustering
from graph import to_bipartite
from mimo import DistributedMIMO, pl_approx_capacity, MassiveMIMO


def topology_spectral():
    n_txs = 128
    n_rxs = 100
    n_clusters = 10
    seed = 9
    mimo = DistributedMIMO(n_txs=n_txs, n_rxs=n_rxs, seed=seed)
    plt.close("all")
    plt.clf()

    # spectral
    labels = spectral(to_bipartite(mimo.pl_matrix_), n_clusters=n_clusters)
    print('Spectral Rate = {0:.3f}'.format(
        pl_approx_capacity(mimo.pl_matrix_, txs_labels=labels[:n_txs], rxs_labels=labels[n_txs:])
    ))
    mimo.draw(txs_labels=labels[:n_txs], rxs_labels=labels[n_txs:], edges=True)

    # co-clustering
    txs_labels, rxs_labels = spectral_co_clustering(mimo.pl_matrix_, n_clusters)
    print('Spectral Co Rate = {0:.3f}'.format(
        pl_approx_capacity(mimo.pl_matrix_, txs_labels=txs_labels, rxs_labels=rxs_labels)
    ))
    mimo.draw(txs_labels=txs_labels, rxs_labels=rxs_labels, edges=True)

    # Affinity Propagation
    af = AffinityPropagation(damping=.9, max_iter=200, convergence_iter=15,
                             copy=True, preference=None, affinity='precomputed'
                             ).fit(X=to_bipartite(mimo.pl_matrix_))
    labels = af.labels_
    print('AP Rate = {0:.3f}'.format(
        pl_approx_capacity(mimo.pl_matrix_, txs_labels=labels[:n_txs], rxs_labels=labels[n_txs:])
    ))
    mimo.draw(txs_labels=labels[:n_txs], rxs_labels=labels[n_txs:], edges=True)

    # # DBSCAN
    # dist = np.ones((n_txs+n_rxs, n_txs+n_rxs)) - to_bipartite(normalize_matrix(mimo.pl_matrix_))
    # db = DBSCAN(eps=1, min_samples=3, metric='precomputed',
    #              metric_params=None, algorithm='auto', leaf_size=30, p=None,
    #              n_jobs=1).fit(X=dist)
    # labels = db.labels_
    # print('DBSCAN Rate = {0:.3f}'.format(
    #     pl_approx_capacity(mimo.pl_matrix_, txs_labels=labels[:n_txs], rxs_labels=labels[n_txs:])
    # ))
    # mimo.draw(txs_labels=labels[:n_txs], rxs_labels=labels[n_txs:], edges=False)

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
    # mimo.rxs_[:, 1] = np.abs(mimo.rxs_[:, 1])
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
    threshold = 9.5

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
        if rate_sp[i] < threshold:
            labels = spectral(to_bipartite(mimo.pl_matrix_), n_clusters=n_clusters)
            txs_labels_sp = labels[:n_txs]
            rxs_labels_sp = labels[n_txs:]
            rate_sp[i] = pl_approx_capacity(mimo.pl_matrix_, txs_labels=txs_labels_sp, rxs_labels=rxs_labels_sp)

        # BS centric
        rxs_labels_bs = fast_update(mimo.pl_matrix_, txs_labels=txs_labels_bs)
        rate_bs[i] = pl_approx_capacity(mimo.pl_matrix_, txs_labels=txs_labels_bs, rxs_labels=rxs_labels_bs)
    plt.figure()
    plt.plot(np.arange(n_slots), rate_sp, 'b-', label='Dynamic Clustering')
    plt.plot(np.arange(n_slots), np.ones(rate_bs.shape) * threshold, 'g-.', label='Threshold')
    plt.plot(np.arange(n_slots), rate_bs, 'k--', label='Static Clustering')
    plt.legend(loc='center right', shadow=True, fontsize='x-large')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1 - 0.3, y2))
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


def rate_beam_clusters():
    n_txs = 16
    seed = 9

    x = np.arange(5, n_txs + 6)
    rate_8 = np.zeros(x.shape)
    rate_10 = np.zeros(x.shape)
    rate_max = np.zeros(x.shape)

    for (n_rxs, cnt) in zip(x, range(x.shape[0])):
        mimo = MassiveMIMO(n_txs=n_txs, n_rxs=n_rxs, seed=seed)

        # max beam association
        txs_labels = np.arange(n_txs)
        rxs_labels = txs_labels[np.argmax(mimo.pl_matrix_, axis=0)]
        rate_max[cnt] = pl_approx_capacity(mimo.pl_matrix_, txs_labels=txs_labels, rxs_labels=rxs_labels)
        n_clusters_beam = count_clusters(txs_labels=txs_labels, rxs_labels=rxs_labels)
        print('target: {0:d}; actual : {1:d}'.format(n_rxs, n_clusters_beam))

        n_clusters = n_clusters_beam - 1
        labels = spectral(to_bipartite(mimo.pl_matrix_), n_clusters=n_clusters)
        rate_8[cnt] = pl_approx_capacity(mimo.pl_matrix_, txs_labels=labels[:n_txs], rxs_labels=labels[n_txs:])
        n_clusters = 10
        labels = spectral(to_bipartite(mimo.pl_matrix_), n_clusters=n_clusters)
        rate_10[cnt] = pl_approx_capacity(mimo.pl_matrix_, txs_labels=labels[:n_txs], rxs_labels=labels[n_txs:])

    plt.figure()
    plt.plot(x, rate_8, 'b-', label='Dynamic Clustering')
    # plt.plot(x, rate_10, 'g-.', label='Dynamic Clustering (10)')
    plt.plot(x, rate_max, 'k--', markersize=10, label='Max Beam')
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
    n_rxs = 16
    n_clusters = 8
    seed = 10
    threshold = 4.0

    n_slots = 50
    rate_sp = np.zeros((n_slots,))
    rate_beam = np.zeros((n_slots,))

    # Initial spectral
    mimo = MassiveMIMO(n_txs=n_txs, n_rxs=n_rxs, seed=seed)
    labels = spectral(to_bipartite(mimo.pl_matrix_), n_clusters=n_clusters)
    txs_labels_sp = labels[:n_txs]

    # max beam association
    txs_labels_beam = np.arange(n_txs)

    # Update
    std = 0.001
    for i in range(n_slots):
        mimo.move(std=std)

        # spectral
        rxs_labels_sp = fast_update(mimo.pl_matrix_, txs_labels=txs_labels_sp)
        rate_sp[i] = pl_approx_capacity(mimo.pl_matrix_, txs_labels=txs_labels_sp, rxs_labels=rxs_labels_sp)
        if rate_sp[i] < threshold:
            labels = spectral(to_bipartite(mimo.pl_matrix_), n_clusters=n_clusters)
            txs_labels_sp = labels[:n_txs]
            rxs_labels_sp = labels[n_txs:]
            rate_sp[i] = pl_approx_capacity(mimo.pl_matrix_, txs_labels=txs_labels_sp, rxs_labels=rxs_labels_sp)

        # max beam
        rxs_labels_beam = fast_update(mimo.pl_matrix_, txs_labels=txs_labels_beam)
        rate_beam[i] = pl_approx_capacity(mimo.pl_matrix_, txs_labels=txs_labels_beam, rxs_labels=rxs_labels_beam)
    plt.figure()
    plt.plot(np.arange(n_slots), rate_sp, 'b-', label='Dynamic Clustering')
    plt.plot(np.arange(n_slots), np.ones(rate_sp.shape) * threshold, 'g-.', label='Threshold')
    plt.plot(np.arange(n_slots), rate_beam, 'k--', label='Max Beam')
    plt.legend(loc='lower right', shadow=True, fontsize='x-large')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1 - 1, y2))
    plt.show()


if __name__ == "__main__":
    # topology_spectral_max_beam()
    # rate_fast_beam()
    # rate_fast()
    # topology_spectral_max_beam()
    # rate_fast_beam()
    rate_beam_clusters()
