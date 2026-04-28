import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

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
    
    df = df.sort_values(by=["point"])

    day_df = df[df["time"] == "day"]
    eve_df = df[df["time"] == "eve"]

    # TODO: Refactor

    # Calculating L_day^c from Brink et al. 2018
    L_day_arr_first = day_df["L_Aeq"].to_numpy()
    L_day_arr_second = eve_df["L_Aeq"].to_numpy()
    
    # 7 for 7 hours from 0700 to 1400
    L_day_arr_first = np.pow(10, L_day_arr_first / 10) * 7
    # 5 for 5 hours from 1400 to 1900
    L_day_arr_second = np.pow(10, L_day_arr_second / 10) * 5

    L_day = 10 * np.log10((L_day_arr_first + L_day_arr_second) / 12)
    L_den = L_day + 1.5

    L_90_arr_first = day_df["L_90"].to_numpy()
    L_10_arr_first = day_df["L_10"].to_numpy()
    L_90_arr_second = eve_df["L_90"].to_numpy()
    L_10_arr_second = eve_df["L_10"].to_numpy()

    L_90_arr_first = np.pow(10, L_90_arr_first / 10) * 7
    L_10_arr_first = np.pow(10, L_10_arr_first / 10) * 7
    L_90_arr_second = np.pow(10, L_90_arr_second / 10) * 7
    L_10_arr_second = np.pow(10, L_10_arr_second / 10) * 7

    L_90 = 10 * np.log10((L_90_arr_first + L_90_arr_second) / 12)
    L_10 = 10 * np.log10((L_10_arr_first + L_10_arr_second) / 12)


    L_TNI = 4 * (L_10 - L_90) + L_90 - 30

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
