"""
This class preprocesses the Credit Card Customer data that are already in the csv file 'data.csv'. More specifically:
    Prints a summary of the actual data
    Normalises the data of the columns that contains useful information and not IDs
    Fills missing values if any exists
    Drops the columns that do not contain useful information but IDs
    Saves the data to a new file 'preprocessed_data.csv'
"""

import pandas as pd
from typing import List

class CreditCardDataPreProcessor:
    def __init__(self, file_path:str):
        """
        Constructor
        :param file_path: strinf that contains the input csv file that we will preprocess
        """

        self.file_path = file_path
        self.data = None

    def load_data(self):
        """
        Loads the CSV file into a pandas DataFrame.
        :return: nothing
        """

        self.data = pd.read_csv(self.file_path)
        print("Data loaded successfully.")

    def summarize_data(self):
        """
        Prints a summary of the dataset.
        :return: nothing
        """

        if self.data is not None:
            print(self.data.describe())
        else:
            print("Data is not loaded. Use load_data() first.")

    def normalize_columns(self, columns: List[str]):
        """
        Normalizes the specified columns using z-score scaling.

        This method applies z-score normalization to the given columns of the dataset. Z-score normalization (or standardization)
        transforms the data by subtracting the mean of each column and then dividing by the standard deviation of that column.

        This ensures that the data in each specified column has a mean of 0 and a standard deviation of 1.

        :param columns: A list of column names to normalize. Each column's values will be transformed using z-score scaling.
        :return: nothing
        """

        if self.data is not None:
            self.data[columns] = self.data[columns].apply(lambda x: (x - x.mean()) / x.std(), axis=0)
            print(f"Columns {columns} normalized using z-score.")
        else:
            print("Data is not loaded. Use load_data() first.")

    def fill_missing_values(self):
        """
        Fills missing values in the dataset with the median of each column.
        :return: nothing
        """

        if self.data is not None:
            self.data.fillna(self.data.median(), inplace=True)
            print("Missing values filled with median values.")
        else:
            print("Data is not loaded. Use load_data() first.")

    def drop_id_columns(self):
        """
        Drops the specified columns by name or index
        :return: nothing
        """

        columns_to_drop = ["Sl_No", "Customer Key"] #we want to drop the IDs columns that we don't need
        self.data = self.data.drop(columns=columns_to_drop)
        print("IDs columns are dropped")


    def save_data(self, output_path:str):
        """
        Saves the processed dataset to a new CSV file.
        :param output_path: string that contains the path to the file that we will store the preprocessed data
        :return: nothing
        """

        if self.data is not None:
            self.data.to_csv(output_path, index=False)
            #print(f"Data saved to {output_path}.")
        else:
            print("Data is not loaded. Use load_data() first.")
