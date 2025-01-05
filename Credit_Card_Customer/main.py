import read_data
import preprocess
import pca_method
import plot
import add_outliers
import hierarchical_clustering
import kMeans
import find_outliers



##################################################################################################################



"""
Αρχικά διαβάζουμε τα δεδομένα από το αρχικό αρχείο txt και τα βάζουμε σε ένα csv αρχείο για ευκολότερη επεξεργασία. 
        
Αυτό γίνεται με τη κλάση DataProcessor στο αρχείο 'read_data.py'
"""

input_file = 'DataInTxt'  # Path to the input text file
output_file = 'data.csv'  # Path to the output CSV file

reader = read_data.DataProcessor(input_file, output_file)
reader.process()

del reader




##################################################################################################################



""" 
Στη συνέχεια κάνουμε μια προ επεξεργασία στα δεδομένα μας.
 
Αυτό γίνεται μέσω της κλάσης CreditCardDataProcessor στο αρχείο 'preprocess.py'
"""

preprocessor = preprocess.CreditCardDataPreProcessor("data.csv")
preprocessor.load_data()
preprocessor.summarize_data()
preprocessor.normalize_columns(["Avg_Credit_Limit", "Total_Credit_Cards", "Total_visits_bank", "Total_visits_online", "Total_calls_made"])
preprocessor.fill_missing_values()
preprocessor.drop_id_columns()
preprocessor.save_data("preprocessed_data.csv")

del preprocessor



##################################################################################################################



"""
Και εδώ εφαρμόζουμε τη μέθοδο PCA για να ρίξουμε τις διαστάσεις σε 2.
 
Αυτό γίνεται μέσω της κλάσης PCAProcessor στο αρχείο 'pca_method.py'
"""

# Initialize the PCAProcessor class with 2 components
pca = pca_method.PCAProcessor(n_components=2)

# Perform PCA
result = pca.fit_transform(
    file_path="preprocessed_data.csv",
    output_path="pca_reduced_data.csv"  # Output file path
)


info = pca.get_information_conserved()
print("Information conserved by PCA: {:.2f}%".format(info * 100))

del result



##################################################################################################################



"""
Εδώ παίρνουμε τα δεδομένα που δημιουργήθηκαν μετά την εφαρμογή του PCA, και τα πλοτάρουμε. 

Αυτό γίνεται με τη βοήθεια της κλάσης CSV2DPlot στο αρχείο 'plot.py'
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
Θα δημιουργήσουμε κάποια outliers και θα δημιουργήσουμε ένα νέο αρχείο μαζί με αυτά. 

Αυτό γίνεται με τη βοήθεια της κλάσης OutlierHandler στο αρχείο 'add_outliers.py'
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



######################################################################################################################



"""
Θα τρέξουμε πρώτα τον ιεραρχικό αλγόριθμο ο οποίος θα μας βρει τον αριθμό με τον οποίο είναι καλύτερο να χωρίσουμε 
τις ομάδες μας.

Αυτό γίνεται με τη βοήθεια της κλάσης HierarchicalClustering που βρίσκεται στο αρχείο 'hierarchical_clustering.py'
"""

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



######################################################################################################################



"""
Τώρα τρέχουμε k-means βάζοντας ως είσοδο k τον αριθμό που βρήκε ο ιεραρχικός.

Αυτό γίνεται με τη βοήθεια της κλάσης KMeans που βρίσκεται στο αρχείο 'kMeans.py'
"""

# Initialize the class
a = kMeans.KMeans(optimal_clusters, file_path="output_with_outliers.csv")

# Run the k-means algorithm
a.run(100)      # Μέγιστο πλήθος επαναλήψεων να είναι το 100. Ο αλγόριθμος ίσως να μη συγκλίνει ακριβώς

# Plot the results
a.plot_results()



######################################################################################################################



"""
Βρίσκουμε τα outliers. Αυτό το κάνουμε ουσιαστικά βρίσκοντας τα πιο απομακρυσμένα από τα κέντρα τους σημεία 
Αυτό γίνεται με βάση τα αποτελέσματα το k-means που έτρεξε προηγουμένως

Αυτό γίνεται με τη βοήθεια της κλάσης OutlierDetector που βρίσκεται στο αρχείο 'find_outliers.py'
"""

# Initialise the class
b = find_outliers.OutlierDetector(optimal_clusters, outlier_percentile=99, file_path="output_with_outliers.csv")

# Find the outliers using the centroids found by the k-means algorithm
b.run(a.centroids)

del a
del b