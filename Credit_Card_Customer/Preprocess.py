import pandas as pd

class CreditCardDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Loads the CSV file into a pandas DataFrame."""
        self.data = pd.read_csv(self.file_path)
        print("Data loaded successfully.")

    def summarize_data(self):
        """Prints a summary of the dataset."""
        if self.data is not None:
            print(self.data.describe())
        else:
            print("Data is not loaded. Use load_data() first.")

    def normalize_columns(self, columns):
        """Normalizes the specified columns using z-score scaling."""
        if self.data is not None:
            self.data[columns] = self.data[columns].apply(lambda x: (x - x.mean()) / x.std(), axis=0)
            print(f"Columns {columns} normalized using z-score.")
        else:
            print("Data is not loaded. Use load_data() first.")

    def fill_missing_values(self):
        """Fills missing values in the dataset with the median of each column."""
        if self.data is not None:
            self.data.fillna(self.data.median(), inplace=True)
            print("Missing values filled with median values.")
        else:
            print("Data is not loaded. Use load_data() first.")

    def drop_id_columns(self):
        """Drop the specified columns by name or index"""
        columns_to_drop = ["Sl_No", "Customer Key"] #we want to drop the IDs columns that we don't need
        self.data = self.data.drop(columns=columns_to_drop)


    def save_data(self, output_path):
        """Saves the processed dataset to a new CSV file."""
        if self.data is not None:
            self.data.to_csv(output_path, index=False)
            print(f"Data saved to {output_path}.")
        else:
            print("Data is not loaded. Use load_data() first.")














########################################################################################################################
#  def detect_outliers(self, column, threshold=1.5):
#     """Detects outliers in a specified column using the IQR method."""
#    if self.data is not None:
#       Q1 = self.data[column].quantile(0.25)
#      Q3 = self.data[column].quantile(0.75)
#     IQR = Q3 - Q1
#    lower_bound = Q1 - threshold * IQR
#   upper_bound = Q3 + threshold * IQR
#  outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
# print(f"Detected {len(outliers)} outliers in column {column}.")
# return outliers
# else:
#   print("Data is not loaded. Use load_data() first.")
#  return None


# def add_outliers(self, column, values):
#    """Adds outlier rows to the dataset for a specific column."""
#    if self.data is not None:
#        for value in values:
#            outlier_row = self.data.iloc[0].copy()
#            outlier_row[column] = value
#            self.data = self.data.append(outlier_row, ignore_index=True)
#        print(f"Added {len(values)} outliers to column {column}.")
#    else:
#        print("Data is not loaded. Use load_data() first.")