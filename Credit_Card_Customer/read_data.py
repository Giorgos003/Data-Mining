import csv
from typing import List

class DataProcessor:
    def __init__(self, input_file:str, output_file:str):
        self.input_file = input_file
        self.output_file = output_file

    def read_data(self):
        with open(self.input_file, 'r') as file:
            # Read all lines in the text file
            lines = file.readlines()

        # The first line is the header, which we will keep
        header = lines[0].strip()

        # The rest of the lines are the actual data
        data = [line.strip() for line in lines[1:]]

        return header, data

    def save_to_csv(self, header: List[str], data: List[str]):
        # Write the data to a CSV file
        with open(self.output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Writing the header
            writer.writerow(header.split(','))
            # Writing the data
            for line in data:
                writer.writerow(line.split(','))

    def process(self):
        header, data = self.read_data()
        self.save_to_csv(header, data)
        print(f"Data has been successfully saved to {self.output_file}")