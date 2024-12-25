import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class OutlierHandler:
    def __init__(self, file_path, x_column, y_column):
        """
        Initialize the class with the file path and column names.

        :param file_path: Path to the CSV file.
        :param x_column: Name of the column to use as X-axis.
        :param y_column: Name of the column to use as Y-axis.
        """
        self.file_path = file_path
        self.x_column = x_column
        self.y_column = y_column
        self.data = None

    def load_data(self):
        """Load the CSV file into a DataFrame."""
        try:
            self.data = pd.read_csv(self.file_path)
            if self.x_column not in self.data.columns or self.y_column not in self.data.columns:
                raise ValueError("Specified columns not found in the CSV file.")
        except Exception as e:
            print(f"Error loading file: {e}")
            self.data = None

    def add_outliers(self, num_outliers=5, multiplier=3):
        """
        Add synthetic outliers to the dataset.

        :param num_outliers: Number of outliers to add.
        :param multiplier: Multiplier to determine the range of outliers.
        """
        if self.data is None:
            print("Data not loaded. Please run `load_data()` first.")
            return

        x_mean, x_std = self.data[self.x_column].mean(), self.data[self.x_column].std()
        y_mean, y_std = self.data[self.y_column].mean(), self.data[self.y_column].std()

        # Generate outliers far from the data range
        outliers_x = np.random.uniform(x_mean + multiplier * x_std, x_mean - multiplier * x_std, num_outliers)
        outliers_y = np.random.uniform(y_mean + multiplier * y_std, y_mean - multiplier * y_std, num_outliers)

        # Append outliers to the DataFrame
        outliers = pd.DataFrame({self.x_column: outliers_x, self.y_column: outliers_y})
        self.data = pd.concat([self.data, outliers], ignore_index=True)

    def plot_data(self):
        """Plot the data including outliers."""
        if self.data is None:
            print("Data not loaded. Please run `load_data()` first.")
            return

        plt.figure(figsize=(8, 6))
        plt.scatter(self.data[self.x_column], self.data[self.y_column], color='blue', alpha=0.7)
        plt.title('2D Scatter Plot with Outliers')
        plt.xlabel(self.x_column)
        plt.ylabel(self.y_column)
        plt.grid(True)
        plt.show()

    def save_data(self, output_path):
        """Save the dataset (with outliers) to a new CSV file."""
        if self.data is not None:
            self.data.to_csv(output_path, index=False)
            print(f"Data saved to {output_path}")
        else:
            print("Data not available. Please load the data and add outliers first.")



