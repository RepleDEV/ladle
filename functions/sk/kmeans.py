import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans

def run(data: np.ndarray):
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(data)

    h = 0.02

    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max()
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap="viridis",
        aspect="auto",
        origin="lower",
    )

    plt.plot(data[:, 0], data[:, 1], "k.", markersize=4)

    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    plt.title(
        "K-means clustering on the digits dataset (PCA-reduced data)\n"
        "Centroids are marked with white cross"
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
