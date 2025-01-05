"""
This class implements the Hierarchical Clustering algorithm.

It takes a linkage criterion and a distance function as input and performs clustering on the dataset.
It also provides a method to find the optimal number of clusters using the silhouette score.
In the end, it plots the data with each cluster having a different colour.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score

class HierarchicalClustering:
    def __init__(self,linkage_criterion='average'):
        """
        Initialize the ClusteringProcessor class.

        Parameters:
        linkage_criterion (str): The linkage criterion to use.
        distance_function (str): The distance function to use.
        """
        self.linkage_criterion = linkage_criterion

    def load_data(self, file_path:str):
        """
        Load the dataset from a CSV file.

        Parameters:
        file_path (str): Path to the input CSV file.
        """
        self.data = pd.read_csv(file_path)
        #print(f"Data loaded successfully from {file_path}")


    def run_hierarchical_clustering(self):
        """
        Perform clustering on the dataset and get the labeled data.
        """
        if self.data is None:
            print("Data not loaded. Please run `load_data()` first.")
            return
        
        linkage_matrix = linkage(self.data, method=self.linkage_criterion, metric='euclidean')
        self.linkage_matrix = linkage_matrix

    
    def find_optimal_clusters(self, max_clusters=10):
        """
        Find the optimal number of clusters using the silhouette score.

        Parameters:
        max_clusters (int): The maximum number of clusters to consider.

        Returns:
        int: The optimal number of clusters.
        """
        if self.data is None:
            print("Data not loaded. Please run `load_data()` first.")
            return

        best_score = -1
        best_clusters = 2
        best_labels = None

        for n_clusters in range(2, max_clusters+1):
            labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
            score = silhouette_score(self.data, labels)
            

            if score > best_score:
                best_score = score
                best_clusters = n_clusters
                best_labels = labels

        print("Silhouette Score:", best_score)
        self.data['Cluster'] = best_labels
        return best_clusters
    
    def save_data(self, output_path):
        """Save the dataset with labesl to a CSV file."""
        if self.data is not None:
            self.data.to_csv(output_path, index=False)
            print(f"Data saved to {output_path}")
        else:
            print("Data not available. Please load the data first")

    def plot_data(self, x_column:str, y_column:str):
        self.x_column = x_column
        self.y_column = y_column
        """Plot the data including cluster labels."""
        if self.data is None:
            print("Data not loaded. Please run `load_data()` first.")
            return

        if 'Cluster' not in self.data.columns:
            print("Cluster labels not found in the data. Ensure clustering is complete.")
            return

        plt.figure(figsize=(8, 6))

        # Use cluster labels to group points and assign colors
        unique_clusters = self.data['Cluster'].unique()
        for cluster in unique_clusters:
            cluster_data = self.data[self.data['Cluster'] == cluster]
            plt.scatter(cluster_data[self.x_column], cluster_data[self.y_column], label=f"Cluster {cluster}", alpha=0.7)

        plt.title('2D Scatter Plot with the clusters suggested by the hierarchical algorithm')
        plt.xlabel(self.x_column)
        plt.ylabel(self.y_column)
        plt.legend()
        plt.grid(True)
        plt.show()
