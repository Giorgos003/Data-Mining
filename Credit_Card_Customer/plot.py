"""
This class plots the data after tha dimensionality reduction.
It loads the data from the 'pca_reduced_data' and plots them
"""

import pandas as pd
import matplotlib.pyplot as plt


class CSV2DPlot:
    def __init__(self, file_path, x_column, y_column):
        """
        Constructor

        :param file_path: Path to the CSV file.
        :param x_column: Name of the column to use as X-axis.
        :param y_column: Name of the column to use as Y-axis.
        """
        self.file_path = file_path
        self.x_column = x_column
        self.y_column = y_column
        self.data = None

    def load_data(self):
        """
        Load the CSV file into a DataFrame.
        :return: nothing
        """
        try:
            self.data = pd.read_csv(self.file_path)
            if self.x_column not in self.data.columns or self.y_column not in self.data.columns:
                raise ValueError("Specified columns not found in the CSV file.")
        except Exception as e:
            print(f"Error loading file: {e}")
            self.data = None

    def plot_data(self):
        """
        Plot the data as a 2D scatter plot.
        :return: nothing
        """
        if self.data is None:
            print("Data not loaded. Please run `load_data()` first.")
            return

        x = self.data[self.x_column]
        y = self.data[self.y_column]

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, color='blue', alpha=0.7)
        plt.title('2D Scatter Plot of CSV Data')
        plt.xlabel(self.x_column)
        plt.ylabel(self.y_column)
        plt.grid(True)
        plt.show()
