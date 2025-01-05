"""
This class finds the outliers that were added at the 'add_outliers.py' file.
It uses the centroids of the clusters found by the k-means algorithm.

At the end, plot data with the possible outliers having a red colour
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

class OutlierDetector:
    def __init__(self, clusters: int, outlier_percentile: float, file_path: str):
        """
        Constructor
        :param clusters: int that contains the number of clusters for K-Means.
        :param outlier_percentile: float that contains percentile threshold to identify outliers.
        :param file_path: string that contains the file that the data are stored
        """

        self.clusters = clusters
        self.outlier_percentile = outlier_percentile
        self.data = []
        self.file_path = file_path

    def load_data(self):
        """
        Loads data from the scv file 'output_with_outliers'
        :return: nothing
        """

        with open(self.file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                self.data.append([float(row[0]), float(row[1])])

    def outliers_detection(self, centroids):
        """
        Detect outliers based on the distances to the centroids.
        :param centroids: List of cluster centroids
        :return: List of outlier indices.
        """

        if not self.data:
            raise ValueError("Data is empty. Load data before trying to detect outliers")

        if not centroids:
            raise ValueError("Centroids are empty. Trying running k-means to detect the centroids before")


        # Compute distances to nearest centroid
        distances = []
        for point in self.data:
            distance_to_centroids = [np.linalg.norm(np.array(point) - np.array(centroid)) for centroid in centroids]    # Calculates the Euclidean distance of every point with every centroid
            distances.append(min(distance_to_centroids))    # It keeps only the distance to the closed centroid

        # Determine the threshold for outliers
        threshold = np.percentile(distances, self.outlier_percentile)

        # Identify outliers
        # The points with distance to the closest centroids greater than the threshold are considered outliers
        outliers = [i for i, distance in enumerate(distances) if distance > threshold]

        return outliers

    def split_the_data(self, centroids):
        """
        Use precomputed centroids from the k-means algorithm to detect outliers. It splits the data to inliers, outliers and centroids
        :param centroids: List that contains precomputed centroids
        :return: Tuple of (inliers, outliers, centroids)
        """

        if not self.data:
            raise ValueError("Data is empty. Load data before processing.")

        # Detect outliers and separates them from the inliers
        outliers = self.outliers_detection(centroids)
        inliers = [i for i in range(len(self.data)) if i not in outliers]

        return inliers, outliers, centroids

    def plot(self, inliers, outliers, centroids):
        """
        Plot the data, highlighting inliers and outliers.
        :param inliers: List of inlier indices.
        :param outliers: List of outlier indices.
        :param centroids: List of cluster centroids.
        :return: nothing
        """

        X = np.array(self.data)
        plt.figure(figsize=(8, 6))

        # Plot inliers
        plt.scatter(X[inliers, 0], X[inliers, 1], c='blue', label='Inliers', alpha=0.7)

        # Plot outliers
        plt.scatter(X[outliers, 0], X[outliers, 1], c='red', label='Outliers', edgecolor='black', s=100)

        # Plot centroids
        plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], c='yellow', label='Centroids', marker='X',
                    s=200)

        # Add labels and legend
        plt.title('K-Means Clustering with Outliers')
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.legend()
        plt.grid(True)
        plt.show()

    def run(self, centroids):
        """
        Runs the class methods
        :param centroids: List that contains the centroid points that the k-means algorithm found
        :return: nothing
        """

        self.load_data()
        inliers, outliers, centroids = self.split_the_data(centroids)
        self.plot(inliers, outliers, centroids)