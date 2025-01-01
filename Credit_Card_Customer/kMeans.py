import csv
import random
import math
from matplotlib import pyplot as plt


def euclidean_distance(point1, point2):
    """
    Calculates the Euclidean distance between two points
    :param point1: a 2-D point
    :param point2: a 2-D point
    :return: the Euclidean distance between point1 and point2
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


class KMeans:
    def __init__(self, k, file_path):
        """
        Constructor
        :param k: the variable k that contains the number of clusters
        :param file_path: the file path that contains the data where we will run the k-means algorithm
        """
        self.k = k
        self.file_path = file_path
        self.data = []
        self.centroids = []
        self.clusters = []

    def load_data(self):
        """
        Loads data from csv file
        :return: nothing
        """
        with open(self.file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                self.data.append([float(row[0]), float(row[1])])

    def initialise_centroids(self):
        """
        Randomly initialises centroids
        :return: nothing
        """
        self.centroids = random.sample(self.data, self.k)

    def assign_clusters(self):
        """
        Assign each point to the nearest centroid
        :return: nothing
        """
        self.clusters = [[] for _ in range(self.k)]
        for point in self.data:
            distances = [euclidean_distance(point, centroid) for centroid in self.centroids]
            cluster_index = distances.index(min(distances))
            self.clusters[cluster_index].append(point)

    def update_centroids(self):
        """
        Updates centroids to the mean of their clusters
        :return: nothing
        """
        for i, cluster in enumerate(self.clusters):
            if cluster:
                x_coords = [point[0] for point in cluster]
                y_coords = [point[1] for point in cluster]
                self.centroids[i] = [sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)]

    def has_converged(self, old_centroids):
        """
        Checks if the algorithm has converged. This means that the old centroids are not updated.
        :return: boolean value indicating if the algorithm has converged or not
        """
        return old_centroids == self.centroids

    def run(self, max_iterations):
        """
        Runs the k-means algorithm
        :return: nothing
        """
        self.load_data()
        self.initialise_centroids()

        for i in range(max_iterations):
            old_centroids = self.centroids[:]
            self.assign_clusters()
            self.update_centroids()

            if self.has_converged(old_centroids):
                break

    def plot_results(self):
        """
        Plots the data points and centroids, coloring each cluster differently.
        :return: nothing
        """
        colors = plt.colormaps["tab10"]  # Updated to use the newer colormaps API
        for i, cluster in enumerate(self.clusters):
            x_coords = [point[0] for point in cluster]
            y_coords = [point[1] for point in cluster]
            plt.scatter(x_coords, y_coords, label=f"Cluster {i + 1}", color=colors(i))

        # Plot centroids
        centroid_x = [centroid[0] for centroid in self.centroids]
        centroid_y = [centroid[1] for centroid in self.centroids]
        plt.scatter(centroid_x, centroid_y, s=200, c="black", marker="X", label="Centroids")

        plt.title("K-Means Clustering")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()
        plt.show()

a = KMeans(k=3, file_path="output_with_outliers.csv")
a.run(100)
a.plot_results()