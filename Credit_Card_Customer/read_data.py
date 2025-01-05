"""
This class reads the data from the txt file ('DataInTxt.txt') and saves them in a csv file ('data.csv') where we can
extract the information much more easily
"""

import csv
from typing import List

class DataProcessor:
    def __init__(self, input_file:str, output_file:str):
        """
        Constructor
        :param input_file: string that contains the txt file where we will read the data from
        :param output_file: string that contains the final csv file
        """
        self.input_file = input_file
        self.output_file = output_file

    def read_data(self):
        """
        Reads data from a specified text file.

        This method opens a text file, reads its content, and splits it into the header and data sections.
        The first line of the file is treated as the header, while the subsequent lines are considered the actual data.
        Both the header and data are returned separately.

        :return header (string): The first line of the file, typically containing column names or field names.
        :return: data (List[string]): A list of the remaining lines from the file, which represent the data entries.
        """

        with open(self.input_file, 'r') as file:
            # Read all lines in the text file
            lines = file.readlines()

        # The first line is the header, which we will keep
        header = lines[0].strip()

        # The rest of the lines are the actual data
        data = [line.strip() for line in lines[1:]]

        return header, data

    def save_to_csv(self, header: str, data: List[str]):
        """
        Writes the data to a CSV file
        :param header: string that contains the column names or field names.
        :param data: A list that contains the data entries.
        :return: nothing
        """
        with open(self.output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Writing the header
            writer.writerow(header.split(','))
            # Writing the data
            for line in data:
                writer.writerow(line.split(','))

    def process(self):
        """
        Runs the above methods so we can store the data to a csv file
        :return: nothing
        """
        header, data = self.read_data()
        self.save_to_csv(header, data)
        print(f"Data has been successfully saved to {self.output_file}")