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
def kmeans(k: int, points: np.ndarray):
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
            # print("\nDone")
            return clusters
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

SCATTER_CLUSTERS_LIST_COLORS = ["red", "royalblue", "limegreen", "gold", "cyan", "fuchsia", "sandybrown", "blueviolet"]

import math
from pandas import DataFrame
def run_kmeans(df: DataFrame):
    print(f"Running Kmeans.")

    day_df = df[df["time"] == "day"]
    eve_df = df[df["time"] == "eve"]

    L_Aeq_day = day_df["L_Aeq"].to_numpy()
    L_Aeq_eve = eve_df["L_Aeq"].to_numpy()

    L_den = soundf.getL_den(L_Aeq_day, L_Aeq_eve)

    L_90_day = day_df["L_90"].to_numpy()
    L_90_eve = eve_df["L_90"].to_numpy()
    L_10_day = day_df["L_10"].to_numpy()
    L_10_eve = eve_df["L_10"].to_numpy()

    L_TNI = soundf.getL_TNI(L_90_day, L_90_eve, L_10_day, L_10_eve)

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
            clusters = kmeans(int(k), points)
            tot_var = getTotalVariance(clusters)
            if tot_var < min_var:
                min_var = tot_var
                min_clusters = clusters

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
    [x, y] = flattened_clusters.transpose() # [L_den, L_TNI]
    cluster_colors = []
    for i, c in enumerate(chosen_clusters):
        cluster_colors.extend([SCATTER_CLUSTERS_LIST_COLORS[i]] * len(c))

    plt.scatter(x, y, c=cluster_colors)
    plt.show()
