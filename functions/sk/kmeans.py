import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from ..kmeans import plotClusters

def run(data: np.ndarray, point_indexes):
    kmeans = KMeans(n_clusters=5)
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
    plotClusters(data, cluster_indexes, point_indexes)
    plt.show()
