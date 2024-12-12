# This class takes as input the preprocessed data and reduces the dimensions from 5 to 2
# The first two columns, that are IDs, are removed from the start

import pandas as pd
from sklearn.decomposition import PCA


class PCAProcessor:
    def __init__(self, n_components=2):
        """
        Initialize the PCAProcessor class.

        Parameters:
        n_components (int): Number of principal components to retain.
        """
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)

    def fit_transform(self, file_path, output_path=None):
        """
        Perform PCA on the dataset and reduce dimensions.

        Parameters:
        file_path (str): Path to the input CSV file.
        output_path (str, optional): Path to save the reduced dataset as a CSV file.

        Returns:
        pd.DataFrame: DataFrame containing reduced dimensions.
        """
        # Load the dataset
        data = pd.read_csv(file_path)

        # Apply PCA to reduce dimensions
        reduced_features = self.pca.fit_transform(data)

        # Create a DataFrame for the PCA results
        reduced_df = pd.DataFrame(reduced_features, columns=[f'PCA{i + 1}' for i in range(self.n_components)])

        # Save the reduced data to a CSV file if output_path is provided
        if output_path:
            reduced_df.to_csv(output_path, index=False)

        return reduced_df


# Example usage
if __name__ == "__main__":
    # Initialize the PCAProcessor class with 2 components
    processor = PCAProcessor(n_components=2)

    # Perform PCA
    result = processor.fit_transform(
        file_path="preprocessed_data.csv",  # Replace with your file path
        output_path="pca_reduced_data.csv"  # Optional output file path
    )

    # Display the result
    print(result.head())