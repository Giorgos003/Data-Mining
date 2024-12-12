import readData
import Preprocess
import pca_method

"""
Αρχικά διαβάζουμε τα δεδομένα από το αρχικό αρχείο txt 
        και τα βάζουμε σε ένα csv αρχείο για ευκολότερη επεξεργασία
"""

input_file = 'DataInTxt'  # Path to the input text file
output_file = 'data.csv'  # Path to the output CSV file

reader = readData.DataProcessor(input_file, output_file)
reader.process()


""" 
Στη συνέχεια κάνουμε μια προ επεξεργασία στα δεδομένα μας μέσω της κλάσης CreditCardDataProcessor 
"""

preprocessor = Preprocess.CreditCardDataProcessor("data.csv")
preprocessor.load_data()
preprocessor.summarize_data()
preprocessor.normalize_columns(["Avg_Credit_Limit", "Total_Credit_Cards", "Total_visits_bank", "Total_visits_online", "Total_calls_made"])
preprocessor.fill_missing_values()
preprocessor.drop_id_columns()
preprocessor.save_data("preprocessed_data.csv")



"""
Και εδώ εφαρμόζουμε τη μέθοδο PCA για να ρίξουμε τις διαστάσεις σε 2
"""

# Initialize the PCAProcessor class with 2 components
pca = pca_method.PCAProcessor(n_components=2)

# Perform PCA
result = pca.fit_transform(
    file_path="preprocessed_data.csv",  # Replace with your file path
    output_path="pca_reduced_data.csv"  # Optional output file path
)

#ΠΡΟΣΟΧΗ, Η ΜΕΤΑΒΛΗΤΗ result ΕΧΕΙ ΑΠΟΘΗΚΕΥΜΕΝΑ ΤΟ ΚΑΙΝΟΥΡΙΟ DATASET ΜΕΤΑ ΤΗΝ ΕΦΑΡΜΟΓΗ ΤΗΣ PCA