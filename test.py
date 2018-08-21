import time

import matplotlib.pyplot as plt
import numpy as np

from cluster import spectral, txs_centric, fast_update
from graph import to_bipartite
from mimo import MIMO, pl_approx_capacity


def plot_clustered_mimo():
    n_txs = 100
    n_rxs = 100
    n_clusters = 10
    seed = 9

    # mimo = ClusteredMIMO(n_txs=n_txs, n_rxs=n_rxs, n_clusters=n_clusters, std=std, seed=seed)
    mimo = MIMO(n_txs=n_txs, n_rxs=n_rxs, seed=seed)
    labels = spectral(to_bipartite(mimo.pl_matrix_), n_clusters=n_clusters)

    plt.close("all")
    plt.clf()
    mimo.draw(txs_labels=labels[:n_txs], rxs_labels=labels[n_txs:], edges=True)
    plt.show()


def compare_capacity(n_txs, n_rxs, n_clusters, seed):
    mimo = MIMO(n_txs=n_txs, n_rxs=n_rxs, seed=seed)

    # Spectral clustering
    labels = spectral(to_bipartite(mimo.pl_matrix_), n_clusters=n_clusters)
    rate_sp = pl_approx_capacity(mimo.pl_matrix_, txs_labels=labels[:n_txs], rxs_labels=labels[n_txs:])

    # BS centric
    txs_labels, rxs_labels = txs_centric(mimo, n_clusters=n_clusters)
    rate_bs = pl_approx_capacity(mimo.pl_matrix_, txs_labels=txs_labels, rxs_labels=rxs_labels)

    return rate_sp, rate_bs


def test_density():
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
            sp, bs = compare_capacity(n_txs=n_txs, n_rxs=n_rxs, n_clusters=n_clusters, seed=5)
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


def plot_fast_control():
    n_txs = 100
    n_rxs = 100
    n_clusters = 10
    seed = 9

    # Initial
    mimo = MIMO(n_txs=n_txs, n_rxs=n_rxs, seed=seed)
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


def test_fast_control():
    n_txs = 100
    n_rxs = 100
    n_clusters = 10
    seed = 9

    n_slots = 50
    rate_sp = np.zeros((n_slots,))
    rate_bs = np.zeros((n_slots,))

    # Initial
    mimo = MIMO(n_txs=n_txs, n_rxs=n_rxs, seed=seed)
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


def test_complexity():
    x = np.arange(50, 501, 50)
    time_sp = np.zeros(x.shape)
    time_bs = np.zeros(x.shape)
    time_fast = np.zeros(x.shape)

    for i in range(x.shape[0]):
        n_clusters = int(x[i] / 10)
        mimo = MIMO(n_txs=x[i], n_rxs=x[i], seed=10)
        t0 = time.time()
        _ = spectral(to_bipartite(mimo.pl_matrix_), n_clusters=n_clusters)
        time_sp[i] = time.time() - t0

        t0 = time.time()
        txs_labels, _ = txs_centric(mimo=mimo, n_clusters=n_clusters)
        time_bs[i] = time.time() - t0

        # Update
        mimo.move(0.001)
        t0 = time.time()
        _ = fast_update(pl_matrix=mimo.pl_matrix_, txs_labels=txs_labels)
        time_fast[i] = time.time() - t0

    print(time_fast)

    fig = plt.figure(0)
    plt.plot(x, time_sp, 'b-', label='Slow channel')
    # plt.plot(x, time_bs, 'k--', label='Static Clustering')
    plt.plot(x, time_fast, 'r-.', label='Fast channel')
    plt.yscale('log')
    plt.legend(loc='best', shadow=True, fontsize='x-large')

    plt.show()


if __name__ == "__main__":
    # plot_clustered_mimo()
    # test_density()
    # test_fast_control()
    test_complexity()
