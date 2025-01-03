import read_data
import preprocess
import pca_method
import plot
import add_outliers
import hierarchical_clustering

##################################################################################################################

"""
Αρχικά διαβάζουμε τα δεδομένα από το αρχικό αρχείο txt 
        και τα βάζουμε σε ένα csv αρχείο για ευκολότερη επεξεργασία
"""

input_file = 'DataInTxt'  # Path to the input text file
output_file = 'data.csv'  # Path to the output CSV file

reader = read_data.DataProcessor(input_file, output_file)
reader.process()

del reader




##################################################################################################################


""" 
Στη συνέχεια κάνουμε μια προ επεξεργασία στα δεδομένα μας μέσω της κλάσης CreditCardDataProcessor 
"""

preprocessor = preprocess.CreditCardDataProcessor("data.csv")
preprocessor.load_data()
preprocessor.summarize_data()
preprocessor.normalize_columns(["Avg_Credit_Limit", "Total_Credit_Cards", "Total_visits_bank", "Total_visits_online", "Total_calls_made"])
preprocessor.fill_missing_values()
preprocessor.drop_id_columns()
preprocessor.save_data("preprocessed_data.csv")

del preprocessor



##################################################################################################################


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


info = pca.get_information_conserved()
print("Information conserved by PCA: {:.2f}%".format(info * 100))

del result


##################################################################################################################


"""
Εδώ παίρνουμε τα δεδομένα που δημιουργήθηκαν μετά την εφαρμογή του PCA, και τα πλοτάρουμε
"""

# Initialize the class
csv_plot = plot.CSV2DPlot('pca_reduced_data.csv', 'PCA1', 'PCA2')  # Replace 'X' and 'Y' with your actual column names

# Load the data
csv_plot.load_data()

# Plot the data
csv_plot.plot_data()

del csv_plot



##################################################################################################################

"""
Θα δημιουργήσουμε κάποια outliers και θα δημιουργήσουμε ένα νέο άρχείο μαζί με αυτά
"""

# Initialize the class
outlier_handler = add_outliers.OutlierHandler('pca_reduced_data.csv', 'PCA1', 'PCA2')

# Load the data
outlier_handler.load_data()

# Add outliers
outlier_handler.add_outliers(num_outliers=5, multiplier=4)

# Plot the data with outliers
outlier_handler.plot_data()

# Save the modified data
outlier_handler.save_data('output_with_outliers.csv')

del outlier_handler

# Initialize the class
hierarchical = hierarchical_clustering.HierarchicalClustering(linkage_criterion='average', distance_function='euclidean')

# Load the data
hierarchical.load_data('output_with_outliers.csv')

# Perform hierarchical clustering
hierarchical.run_hierarchical_clustering()

# Find the optimal number of clusters
"""Αυτό θα χρησιμοποιήσουμε για τον αριθμό των clusters στον k-means"""
optimal_clusters = hierarchical.find_optimal_clusters(max_clusters=10)
print(f"Optimal number of clusters: {optimal_clusters}")

# Save the labeled data
hierarchical.save_data('output_with_clusters.csv')

# Plot the data with cluster labels
hierarchical.plot_data("PCA1", "PCA2")

del hierarchical


