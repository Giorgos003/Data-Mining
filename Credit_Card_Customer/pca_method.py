"""
This class takes as input the preprocessed data from the 'preprocessed_data.csv' file  and reduces the dimensions from 5 to 2.
The first two columns, that are IDs, are removed from the start (check the 'preprocessed.py' file).
It saves the dimensional reduced data to a new file 'pca_reduced_data.csv'

It is also the calculates the percentage of the information that it is conserved after the dimensionality reduction. It prints this value
"""

import pandas as pd
from sklearn.decomposition import PCA


class PCAProcessor:
    def __init__(self, n_components: int):
        """
        Constructor
        :param n_components: integer that contains the number of principal components to retain.
        """

        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)

    def fit_transform(self, file_path: str, output_path: str):
        """
        Performs PCA on the dataset and reduce dimensions.
        :param file_path: string that contains the path to the input CSV file.
        :param output_path: string that contains the path to save the reduced dataset as a CSV file.
        :return: the DataFrame containing reduced dimensions.
        """

        # Load the dataset
        data = pd.read_csv(file_path)

        # Apply PCA to reduce dimensions
        reduced_features = self.pca.fit_transform(data)

        # Create a DataFrame for the PCA results
        reduced_df = pd.DataFrame(reduced_features, columns=[f'PCA{i + 1}' for i in range(self.n_components)])

        # Save the reduced data to a CSV file if output_path is provided
        if output_path is not None:
            reduced_df.to_csv(output_path, index=False)

        return reduced_df
    
    def get_information_conserved(self):
        """
        Calculates the proportion of variance explained by the selected principal components in PCA.
        :return: that number
        """
        return sum(self.pca.explained_variance_ratio_)
