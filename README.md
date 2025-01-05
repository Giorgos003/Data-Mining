# Data-Mining
Εργασία εξαμήνου στο μάθημα Αποθήκες και Εξόρυξη Δεδομένων 2024 - 2025

Τα δεδομένα τα οποία θα χρησιμοποιήσουμε τα πήραμε από το [Kaggle](https://www.kaggle.com/), και είναι το [Credit Card Customer Data](https://www.kaggle.com/datasets/aryashah2k/credit-card-customer-data)

Για να τρέξετε το πρόγραμμα απλώς πατάτε run στη 'main.py'. Προσοχή διότι χρησιμοποιούνται διάφορες βιβλιοθήκες στα διάφορα αρχεία, να είναι ήδη εγκαταστημένες για να τρέξει ο κώδικας.

Μετά το πέρας της εκτέλεσης θα εμφανιστούν ως έξοδο στο τερματικό διάφορα στοιχεία. Είναι πληροφορίες που δείχνουν ορισμένες από τις διαδικασίες που εκτέλεσε ο κώδικας αλλά και διάφορες χρήσιμες πληροφορίες όπως 'πόση πληροφορία χάθηκε εξ αιτίες του dimensionality reduction', τα 'Silhouette Scores' και 'τον optimal number of clusters που βρέθηκε από τον ιεραρχικό αλγόριθμο για να τρέξει μετά ο k-means' 

Επίσης θα εμφανιστούν πέντε (5) γραφήματα
  Το πρώτο αναπαριστά τα δεδομένα αμέσως μετά το dimensionality reduction
  Το δεύτερο αναπαριστά τα δεδομένα μαζί και με τα outliers που προστέθηκαν
  Το τρίτο αναπαριστά τα δεδομένα έτσι όπως προτείνει ο ιεραρχικός αλγόριθμος ότι πρέπει να χωριστούν
  Το τέταρτο αναπαριστά τα δεδομένα μετά την εκτέλεση του αλγορίθμου k-means
  Το πέμπτο και τελευταίο αναπαριστά τα δεδομένα τονίζοντας τα outliers που βρέθηκαν
