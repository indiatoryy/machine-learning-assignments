# K-Means Clustering Algorithm 

## Introduction:
K-Means is a fundamental unsupervised machine learning algorithm used for clustering and data segmentation. The goal of clustering is to group similar data points together, allowing data scientists and researchers to uncover patterns, gain insights, and make data-driven decisions. K-Means achieves this by partitioning a dataset into K clusters, with each cluster represented by its centroid, which is the mean of all data points in the cluster.

## Background:

K-Means clustering is based on the following principles:

* **Euclidean Distance:** K-Means relies on Euclidean distance to measure the dissimilarity between data points. In a multi-dimensional space, Euclidean distance calculates the straight-line distance between two points, forming the basis for clustering similar data points together.
* **Centroid:** A centroid represents the center of a cluster. During each iteration of the algorithm, data points are assigned to the nearest centroid, and the centroids are recalculated as the mean of all data points in the cluster. This iterative process refines the cluster assignments and centroids until convergence.

## Algorithm Overview:
The K-Means algorithm can be summarized into the following steps:

* **Initialization:** K initial centroids are placed randomly in the data space. Alternatively, more sophisticated methods, such as K-Means++ or Forgy, can be used for intelligent initialization.
* **Assignment Step:** Each data point is assigned to the nearest centroid based on Euclidean distance. This step forms clusters based on proximity.
* **Update Step:** The centroids are recalculated as the mean of all data points assigned to each cluster. This step repositions the centroids at the center of their respective clusters.
* **Iteration:** Steps 2 and 3 are repeated iteratively until the centroids converge (i.e., they no longer change significantly) or a predetermined number of iterations are reached.

## Implementation Details:

This implementation of K-Means clustering algorithm is done in Python, utilizing only the NumPy library for numerical computations. The provided function takes in an array representing the dataset and a value κ (number of clusters) as input parameters. It returns the cluster centroids and the cluster assignments for each data point.

### Initialization Heuristic:
Students are required to choose one of the two initialization heuristics discussed in class for initializing the centroids. The initialization heuristic used in this implementation is the random initialization method. In the k_means function, centroids are initialized randomly from the available data points. This random initialization method is a common approach to start the K-means clustering algorithm.

