import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import a hierarchical clustering algorithm
from sklearn.cluster import AgglomerativeClustering

class HierarchicalClustering:
    def __init__(self, n_clusters=3,linkage_criterion='average',distance_function='euclidean'):
        """
        Initialize the ClusteringProcessor class.

        Parameters:
        n_clusters (int): Number of clusters to form.
        linkage_criterion (str): The linkage criterion to use.
        distance_function (str): The distance function to use.
        """
        self.n_clusters = n_clusters
        self.linkage_criterion = linkage_criterion
        self.distance_function = distance_function
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters,linkage=self.linkage_criterion,metric=self.distance_function)

    def load_data(self, file_path:str):
        """
        Load the dataset from a CSV file.

        Parameters:
        file_path (str): Path to the input CSV file.
        """
        self.data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")


    def run_hierarchical_clustering(self):
        """
        Perform clustering on the dataset and get the labeled data.
        """
        if self.data is None:
            print("Data not loaded. Please run `load_data()` first.")
            return
        # Apply clustering and store the labels
        labels = self.model.fit_predict(self.data)

        # Add the labels to the dataset
        self.data['Cluster'] = labels

        return self.data
    
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

        plt.title('2D Scatter Plot with Clusters')
        plt.xlabel(self.x_column)
        plt.ylabel(self.y_column)
        plt.legend()
        plt.grid(True)
        plt.show()
