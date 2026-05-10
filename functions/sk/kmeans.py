import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from ..kmeans import plotClusters

def run(data: np.ndarray, point_indexes):
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(data)

    # h = 0.02
    #
    # x_min, x_max = data[:, 0].min() - 1, data[:, 0].max()
    # y_min, y_max = data[:, 1].min() - 1, data[:, 1].max()
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #
    # Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    #
    # Z = Z.reshape(xx.shape)
    # plt.figure(1)
    # plt.clf()
    # plt.imshow(
    #     Z,
    #     interpolation="nearest",
    #     extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    #     cmap="viridis",
    #     aspect="auto",
    #     origin="lower",
    # )
    
    cluster_indexes = kmeans.labels_

    cluster_L_den = [c[0] for c in kmeans.cluster_centers_]
    cluster_L_den_sort = np.argsort(cluster_L_den)

    print(f"Clusters sorted by L_den: ")
    for i, c_index in enumerate(cluster_L_den_sort):
        print_msg = f"{i + 1} (L_den = {cluster_L_den[c_index]}): "
        points = []
        for j, point_cluster_index in enumerate(cluster_indexes):
            if point_cluster_index == c_index:
                points.append(f"Point {point_indexes[j]}")

        print_msg += ", ".join(points)
        print(print_msg)

    plotClusters(data, cluster_indexes, point_indexes)
    plt.show()
