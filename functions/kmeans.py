"""
Ladle, a noise data analysis program.
Copyright (C) 2026 Ragazzo Chaesa

This file is part of Ladle.

Ladle is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

Ladle is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Ladle. If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import functions.sound as soundf

rng = np.random.default_rng()

KMEANS_MAX_ITERATIONS = 1e5
def kmeans(k: int, points: np.ndarray) -> dict[str, list]:
    centroids = rng.choice(points, k, replace=False) # k random points

    # print(f"Running kmeans iteration. k = {k}, points shape: {points.shape}")
    # print(f"Running with centroids: {centroids}")

    i = 0
    while True:
        clusters = [[] for _ in range(k)]

        # Assign points to nearest cluster
        for p in points:
            centroid_distances = [np.linalg.norm(np.array(p) - np.array(c)) for c in centroids]
            index_min = np.argmin(centroid_distances)


            clusters[index_min].append(p)

        # Calculate centroids / mean
        newCentroids = []

        for i in range(len(clusters)):
            c = clusters[i]
            newCentroid = np.mean(c, axis=0) if len(c) else centroids[i]
            newCentroids.append(newCentroid)

        newCentroids = np.array(newCentroids)

        i += 1

        if i > KMEANS_MAX_ITERATIONS:
            raise RuntimeError(f"Kmeans max iterations reached: {KMEANS_MAX_ITERATIONS}")

        # print(f"Current iteration: {i}", end="\r")

        # Convergence check
        if (newCentroids==centroids).all():
            indexes = []

            # Resolve cluster indexes
            for p in points:
                for i, c in enumerate(clusters):
                    for cluster_point in c:
                        if (cluster_point == p).all():
                            indexes.append(i)
                            break
            result = {
                "clusters": clusters,
                "centroids": centroids,
                "indexes": indexes
            }
            return result
        else:
            centroids = newCentroids

def getTotalVariance(clusters):
    variance = 0

    for c in clusters:
        centroid = np.mean(c, axis=0)
        square_dists = np.sum((np.array(c) - centroid)**2, axis=1)
        var = np.mean(square_dists)
        variance += var

    return variance

def resolvePoints(points: np.ndarray, clusters_points: np.ndarray):
    indexes = []
    for c in clusters_points:
        for i, p in enumerate(points):
            if (c == p).all():
                indexes.append(i)
    return indexes

SCATTER_CLUSTERS_LIST_COLORS = ["red", "royalblue", "limegreen", "gold", "cyan", "fuchsia", "sandybrown", "blueviolet"]

from typing import List
from sklearn.decomposition import PCA
def plotClusters(points: np.ndarray, cluster_indexes: np.ndarray, points_labels: List[str] = []):
    if not len(points_labels):
        range_list = list(range(len(points)))
        points_labels = [f"Point {p + 1}" for p in range_list]

    cluster_colors = [SCATTER_CLUSTERS_LIST_COLORS[i] for i in cluster_indexes]

    # if higher than 2-dimensional data, reduce by PCA
    if (points.shape[1]):
        points = PCA(n_components=2).fit_transform(points)

    for i, [x, y] in enumerate(points): 
        label = points_labels[i]
        plt.text(x + 0.2, y + 0.2, label)

    [x, y] = points.transpose()
    plt.scatter(x,y, c=cluster_colors)
    plt.show()

import math
from pandas import DataFrame
def run_kmeans(L_den: np.ndarray, L_TNI: np.ndarray, point_labels: List[str] = []):
    print(f"Running Kmeans.")

    # Create list of [L_den, L_TNI] vectors of each point
    points = np.vstack((L_den, L_TNI)).transpose()

    RUNS_PER_K = 50

    print(f"Running (max) {RUNS_PER_K} runs per k value. Running k means with k 1 -> 10.")
    min_vars = []
    list_clusters = []

    k_values = np.arange(1, 10 + 1)
    for k in k_values:
        total_runs = math.comb(len(points), k) # nCr
        if total_runs > RUNS_PER_K:
            total_runs = RUNS_PER_K

        min_var = 2147483648 # 32 bit int max
        min_clusters = None
        
        # Run through k means total_runs times, extracting the run with the least total WCSS
        while total_runs > 0:
            kmeans_res = kmeans(int(k), points)
            clusters = kmeans_res["clusters"]
            tot_var = getTotalVariance(clusters)
            if tot_var < min_var:
                min_var = tot_var
                min_clusters = kmeans_res 

            total_runs -= 1
        min_vars.append(min_var)
        list_clusters.append(min_clusters)

    print("Running manual elbow analysis. Close figure to continue.")

    plt.plot(k_values, min_vars)
    plt.show()

    chosen_k = int(input("Choose a k value to continue: "))
    if not (chosen_k in k_values):
        raise ValueError("Invalid k value chosen. Quitting.")

    print("Plotting clusters for chosen k value.")
    chosen_clusters = list_clusters[chosen_k - 1]
    flattened_clusters = np.array([p for c in chosen_clusters for p in c])
    clusters_points = flattened_clusters.transpose() # [L_den, L_TNI]
    clusters_indices = chosen_clusters["indexes"]

    plotClusters(points, clusters_indices)
