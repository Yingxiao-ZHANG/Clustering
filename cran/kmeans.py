# k-means
import random
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from cran.mimo import ClusteredMIMO
from cran.svd import distributed_svd

SHOW = False
FIG = 0
ANCHOR = 2


class Processor:
    def __init__(self, index, features, locations, markers):
        self.index = index
        self.features = features  # list of N-dimension arrays
        self.locations = locations
        self.markers = markers
        self.labels = np.ones((features.shape[0],), dtype='int') * index
        self.local_message = []
        self.neighbors = set()
        self.centroids = None

    def set_centroids(self, centroids):
        self.centroids = centroids
        self.labels = nearest_centroid(features=self.features, centroids=self.centroids)

    def update_centroids(self):
        counts = self.local_message['counts']
        counts[counts == 0] = 1
        self.centroids = np.diag(1 / counts).dot(self.local_message['sums'])
        self.labels = nearest_centroid(features=self.features, centroids=self.centroids)

    def init_message(self):
        global ANCHOR
        if self.index == ANCHOR:
            self.local_message = kmeans_message(features=self.features, centroids=self.centroids, weight=1)
        else:
            self.local_message = kmeans_message(features=self.features, centroids=self.centroids, weight=0)

    def send(self):
        receiver = random.sample(self.neighbors, 1)[0]
        message = dict()
        for key in self.local_message:
            message[key] = self.local_message[key] / 2
            self.local_message[key] -= message[key]
        receiver.receive(sender=self, message=message, reply=True)

    def receive(self, sender, message, reply):
        if reply:
            message_reply = dict()
            for key in self.local_message:
                message_reply[key] = self.local_message[key] / 2
                self.local_message[key] -= message_reply[key]
            sender.receive(sender=self, message=message_reply, reply=False)
        for key in self.local_message:
            self.local_message[key] += message[key]
        # update neighbours
        # self.neighbors.add(sender)
        # self.neighbors.union(sender.neighbors)


def nearest_centroid(features, centroids):
    dist = cdist(features, centroids, 'euclidean')
    return np.argmin(dist, axis=1)


def kmeans_message(features, centroids, weight=0):
    labels = nearest_centroid(features, centroids)
    n_clusters = centroids.shape[0]
    local_sums = np.zeros(centroids.shape)
    local_counts = np.zeros((n_clusters,))
    local_errors = np.zeros((n_clusters,))
    for i in range(n_clusters):
        active = labels == i
        local_counts[i] = np.sum(active)
        if local_counts[i] == 0:
            continue
        local_sums[i] = np.sum(features[active], axis=0)
        local_errors[i] = np.sum(np.square(features[active] - centroids[i]))
    return {'sums': local_sums,
            'counts': local_counts,
            'error': np.sum(local_errors),
            'weight': weight
            }


def gossip(servers, n_iterations=20):
    m = np.zeros((n_iterations,))
    v = np.zeros((n_iterations,))

    for iteration in range(n_iterations):
        for server in servers:
            server.send()
        data = np.asarray([server.local_message['weight'] for server in servers])
        m[iteration] = np.mean(data)
        v[iteration] = np.var(data)
    # global FIG
    # FIG += 1
    # plt.figure(FIG)
    # plt.plot(np.arange(0, n_iterations), m, label='mean')
    # plt.plot(np.arange(0, n_iterations), v, label='var')
    # plt.legend()
    # plt.title('Convergence of weight in gossip protocol, n_txs={0}'.format(len(servers)))
    # plt.xlabel('num. of cycles')
    # plt.ylabel('global statistics')


def init_centroids(servers, n_centroids, seed=3):
    dim_features = servers[0].features.shape[1]
    dim_locations = servers[0].locations.shape[1]
    features = np.zeros((n_centroids, dim_features))
    locations = np.zeros((n_centroids, dim_locations))

    np.random.seed(seed)
    num = np.random.randint(0, 1000, (1,))[0]
    for i in range(n_centroids):
        num = num + 1
        server_id = int(num % len(servers))
        node_id = int(num % servers[server_id].features.shape[0])
        features[i] = servers[server_id].features[node_id]
        locations[i] = servers[server_id].locations[node_id]
    return {'features': features, 'locations': locations}


def gossip_kmeans(servers, n_clusters, n_iterations=10):
    centroids = init_centroids(servers=servers, n_centroids=n_clusters)
    for server in servers:
        server.set_centroids(centroids['features'])
        server.init_message()
    plot_topology(servers)
    plt.plot(centroids['locations'][..., 0], centroids['locations'][..., 1], 'xk')

    iteration = 0
    results = np.zeros((n_iterations,))
    while iteration < n_iterations:
        gossip(servers)
        for i, server in enumerate(servers):
            results[iteration] += server.local_message['error']
            server.update_centroids()
            server.init_message()
        iteration += 1
    plot_topology(servers)
    print('Final centroids:')
    print(servers[0].centroids)
    # plt.plot(servers[0].centroids[..., 0], servers[0].centroids[..., 1], 'xk')

    global FIG
    FIG += 1
    plt.figure(FIG)
    plt.plot(np.arange(0, n_iterations), results.T)
    plt.xlabel('iterations')
    plt.ylabel('global error')
    plt.title('Distributed kmeans: n_txs={0}'.format(len(servers)))


def find_neighbors(index, servers):
    neighbors = set(servers)
    neighbors.remove(servers[index])
    return neighbors


def build_servers(mimo, features):
    n_txs = mimo.txs_.shape[0]
    server_id = np.hstack([np.arange(0, n_txs), np.argmax(mimo.pl_matrix_, axis=0)])
    markers = np.zeros(server_id.shape)
    markers[:n_txs] = 1  # BS = 1, UE = 0
    locations = np.vstack([mimo.txs_, mimo.rxs_])

    # construct server nodes
    servers = []
    for index in range(n_txs):
        selected = server_id == index
        servers.append(Processor(index=index,
                                 features=features[selected],
                                 locations=locations[selected],
                                 markers=markers[selected]
                                 ))

    # connect servers as a fully connected network
    for index in range(n_txs):
        servers[index].neighbors = find_neighbors(index=index, servers=servers)

    return servers


def plot_topology(servers):
    nodes = []
    labels = []
    markers = []
    for server in servers:
        nodes.extend(server.locations)
        labels.extend(server.labels)
        markers.extend(server.markers)
    nodes = np.asarray(nodes)
    labels = np.asarray(labels)
    markers = np.asarray(markers)

    global FIG, SHOW
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    FIG += 1
    plt.figure(FIG)
    for label, color in zip(range(np.max(labels) + 1), colors):
        plt.plot(nodes[np.logical_and(labels == label, markers == 0)][..., 0],
                 nodes[np.logical_and(labels == label, markers == 0)][..., 1],
                 color + 'o', markersize=5, mfc='none')
        plt.plot(nodes[np.logical_and(labels == label, markers == 1)][..., 0],
                 nodes[np.logical_and(labels == label, markers == 1)][..., 1],
                 color + '^', markersize=5, mfc='none')
    SHOW = True


def centralized_kmeans(servers, n_clusters):
    centroids = init_centroids(servers=servers, n_centroids=n_clusters)
    data = []
    for server in servers:
        data.extend(server.features)
    kmeans = KMeans(n_clusters=n_clusters, init=centroids['features'], n_init=1).fit(np.asarray(data))
    for server in servers:
        server.set_centroids(kmeans.cluster_centers_)
    plot_topology(servers)


def init_features(mimo, n_clusters, method='locations'):
    n_txs = mimo.txs_.shape[0]
    n_rxs = mimo.rxs_.shape[0]

    # Calculate symmetric laplacian of the bipartite graph
    adj = np.vstack([np.hstack([np.zeros((n_txs, n_txs)), mimo.pl_matrix_]),
                     np.hstack([mimo.pl_matrix_.T, np.zeros((n_rxs, n_rxs))])
                     ])
    degree = np.sum(adj, axis=1)
    if np.any(degree <= 0):
        raise Exception('exist unconnected nodes')
    degree_sqrt_inverse = np.diag(np.power(degree, - 1 / 2))

    laplacian = np.eye(n_txs + n_rxs) + degree_sqrt_inverse.dot(adj).dot(degree_sqrt_inverse)

    if method == 'numpy svd':
        u, s, vh = npl.svd(laplacian, full_matrices=False, compute_uv=True)
        features = u[..., :n_clusters]
        features = degree_sqrt_inverse.dot(features)
    elif method == 'distributed svd':
        features = distributed_svd(laplacian, rank=n_clusters, n_iter=1000)
        features = degree_sqrt_inverse.dot(features)
        # features = np.diag(1 / npl.norm(features, axis=1)).dot(features)    # row normalization
    else:
        features = np.vstack([mimo.txs_, mimo.rxs_])

    return features


def test(method):
    # construct wireless mimo system
    n_txs = 10
    n_rxs = 15
    n_clusters: int = 3
    np.random.seed(1)
    # mimo = DistributedMIMO(n_txs, n_rxs, 0)
    mimo = ClusteredMIMO(n_txs=n_txs, n_rxs=n_rxs, n_clusters=n_clusters + 1, std=0.1)

    # calculate eigen-vectors
    features = init_features(mimo, n_clusters, method=method)  # locations, numpy svd, distr svd

    # construct gossip network between txs
    servers = build_servers(mimo=mimo, features=features)

    # kmeans to find cluster labels
    gossip_kmeans(servers=servers, n_clusters=n_clusters)
    # labels_centralized = centralized_kmeans(servers=servers, n_clusters=n_clusters)
    # centralized_kmeans(servers, n_clusters)


if __name__ == '__main__':
    test(method='numpy svd')

    if SHOW:
        plt.show()
