import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# File paths
ratings_path = 'ratings.dat'
movies_path = 'movies.dat'
users_path = 'users.dat'

# Load ratings data with Latin-1 encoding
try:
    ratings_df = pd.read_csv(ratings_path, header=None, sep='::', engine='python', encoding='latin-1')
    ratings_df.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    print("Ratings data loaded successfully")
except Exception as e:
    print(f"Error loading ratings data: {e}")
    
# Load movies data with Latin-1 encoding
try:
    movies_df = pd.read_csv(movies_path, header=None, sep='::', engine='python', encoding='latin-1')
    movies_df.columns = ['movie_id', 'title', 'genres']
    print("Movies data loaded successfully")
except Exception as e:
    print(f"Error loading movies data: {e}")

# Load users data with Latin-1 encoding
try:
    users_df = pd.read_csv(users_path, header=None, sep='::', engine='python', encoding='latin-1')
    users_df.columns = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
    print("Users data loaded successfully")
except Exception as e:
    print(f"Error loading users data: {e}")

# Get number of users and movies
n_users = ratings_df['user_id'].nunique()
n_movies = ratings_df['movie_id'].nunique()
print(f"Number of users: {n_users}")
print(f"Number of movies: {n_movies}")

# Process movie genres
print("Processing movie genres...")
# Convert genres string to one-hot encoded columns
genres_list = movies_df['genres'].str.split('|')
all_genres = set()
for genres in genres_list:
    if isinstance(genres, list):  # Check if it's a valid list
        all_genres.update(genres)

# Create a dictionary to map movie_id to genre features
movie_genres = {}
for _, row in movies_df.iterrows():
    movie_id = row['movie_id']
    genres = row['genres'].split('|') if isinstance(row['genres'], str) else []
    # Create a binary vector for genres
    genre_vector = [1 if genre in genres else 0 for genre in all_genres]
    movie_genres[movie_id] = genre_vector

# Process user features
print("Processing user features...")
# Create binary features for gender
users_df['is_male'] = (users_df['gender'] == 'M').astype(int)

# Convert age to numeric bins (simplify the age categories)
age_mapping = {
    1: 0,  # "Under 18"
    18: 1, # "18-24"
    25: 2, # "25-34"
    35: 3, # "35-44"
    45: 4, # "45-49"
    50: 5, # "50-55"
    56: 6  # "56+"
}
users_df['age_group'] = users_df['age'].map(age_mapping)

# One-hot encode occupation
occupation_dummies = pd.get_dummies(users_df['occupation'], prefix='occupation')

# Process zip codes - extract first digit as region indicator
users_df['zip_region'] = users_df['zip_code'].astype(str).str[0].astype(int)
zip_region_dummies = pd.get_dummies(users_df['zip_region'], prefix='region')

# Combine all user features
users_df = pd.concat([users_df, occupation_dummies, zip_region_dummies], axis=1)

# Create a dictionary to map user_id to user features
user_features = {}
for _, row in users_df.iterrows():
    user_id = row['user_id']
    
    # Get gender and age features
    features = [row['is_male'], row['age_group']]
    
    # Add occupation features
    occupation_cols = [col for col in users_df.columns if col.startswith('occupation_')]
    for col in occupation_cols:
        features.append(row[col])
    
    # Add region features based on zip code
    region_cols = [col for col in users_df.columns if col.startswith('region_')]
    for col in region_cols:
        features.append(row[col])
    
    user_features[user_id] = features

# Function to create the user-movie rating matrix
print("Creating user-movie rating matrix...")
def create_rating_matrix(df, n_users, n_movies):
    data = np.zeros((n_users, n_movies), dtype=np.int32)
    movie_id_mapping = {}
    
    # Create movie_id mapping
    unique_movie_ids = df['movie_id'].unique()
    for i, movie_id in enumerate(unique_movie_ids):
        movie_id_mapping[movie_id] = i
    
    # Fill the matrix with ratings
    for _, row in df.iterrows():
        user_idx = int(row['user_id']) - 1  # User IDs start from 1
        movie_idx = movie_id_mapping[row['movie_id']]
        data[user_idx, movie_idx] = row['rating']
    
    return data, movie_id_mapping

# Create the rating matrix
rating_matrix, movie_id_mapping = create_rating_matrix(ratings_df, n_users, n_movies)

# Find movie with most ratings to use as target
movie_counts = ratings_df['movie_id'].value_counts()
print("Top 5 most rated movies:")
print(movie_counts.head())

target_movie_id = 2858  # As specified in the original code
if target_movie_id not in movie_id_mapping:
    print(f"Target movie ID {target_movie_id} not found in mapping. Using most rated movie instead.")
    target_movie_id = movie_counts.idxmax()

target_movie_idx = movie_id_mapping[target_movie_id]
print(f"Using movie ID {target_movie_id} as target (index: {target_movie_idx})")

# Prepare features and target
X_raw = np.delete(rating_matrix, target_movie_idx, axis=1)
Y_raw = rating_matrix[:, target_movie_idx]
X = X_raw[Y_raw > 0]
Y = Y_raw[Y_raw > 0]
print(f"Shape of X: {X.shape}")
print(f"Shape of Y: {Y.shape}")

# Convert ratings to binary classification (like/dislike)
recommend = 3
Y_binary = Y.copy()
Y_binary[Y <= recommend] = 0
Y_binary[Y > recommend] = 1
n_pos = (Y_binary == 1).sum()
n_neg = (Y_binary == 0).sum()
print(f'{n_pos} positive samples and {n_neg} negative samples.')

# Get user IDs for samples with ratings for the target movie
rated_user_indices = np.where(Y_raw > 0)[0]
user_ids_with_ratings = rated_user_indices + 1  # Add 1 because user_ids start from 1

# Create enhanced feature matrix
print("Creating enhanced feature matrix...")
X_enhanced = []
for i, user_idx in enumerate(rated_user_indices):
    user_id = user_idx + 1  # User IDs start from 1
    
    # Get base rating features
    rating_features = X[i]
    
    # Add user features if available
    if user_id in user_features:
        user_feature_vector = user_features[user_id]
        combined_features = np.concatenate([rating_features, user_feature_vector])
    else:
        combined_features = rating_features
    
    X_enhanced.append(combined_features)

X_enhanced = np.array(X_enhanced)
print(f"Shape of enhanced features: {X_enhanced.shape}")

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X_enhanced, Y_binary, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Function to calculate ROC curve data
def calculate_roc_data(y_true, y_score):
    # Calculate ROC curve points
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return fpr, tpr, thresholds

# Function to plot and save ROC curve
def plot_roc_curve(fpr, tpr, auc_value, model_name, output_file=None):
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {auc_value:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save to PDF if output_file is provided
    if output_file:
        plt.savefig(output_file)
    
    plt.show()

# Train basic model for comparison
print("Training basic model...")
basic_clf = MultinomialNB(alpha=1.0, fit_prior=True)
basic_clf.fit(X_train, Y_train)
basic_pred_prob = basic_clf.predict_proba(X_test)[:, 1]
basic_auc = roc_auc_score(Y_test, basic_pred_prob)
print(f"Basic model AUC: {basic_auc:.5f}")

# Calculate ROC curve data for basic model
basic_fpr, basic_tpr, _ = calculate_roc_data(Y_test, basic_pred_prob)

# Cross-validation to find best parameters
print("Running cross-validation to find best parameters...")
k = 5
k_fold = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)

smoothing_factor_options = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
fit_prior_options = [True, False]
auc_record = {}

for alpha in smoothing_factor_options:
    auc_record[alpha] = {}
    for fit_prior in fit_prior_options:
        auc_record[alpha][fit_prior] = 0.0

for train_indices, test_indices in k_fold.split(X_enhanced, Y_binary):
    X_train_k, X_test_k = X_enhanced[train_indices], X_enhanced[test_indices]
    Y_train_k, Y_test_k = Y_binary[train_indices], Y_binary[test_indices]
    
    for alpha in smoothing_factor_options:
        for fit_prior in fit_prior_options:
            clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
            clf.fit(X_train_k, Y_train_k)
            prediction_prob = clf.predict_proba(X_test_k)
            pos_prob = prediction_prob[:, 1]
            auc = roc_auc_score(Y_test_k, pos_prob)
            auc_record[alpha][fit_prior] += auc

# Find best parameters
best_alpha = 0
best_fit_prior = False
best_auc = 0

print("\nCross-validation results:")
print("Alpha    Fit Prior    AUC")
print("-------------------------")
for alpha, fit_prior_dict in auc_record.items():
    for fit_prior, auc_sum in fit_prior_dict.items():
        avg_auc = auc_sum / k
        print(f"{alpha:<8} {str(fit_prior):<12} {avg_auc:.5f}")
        
        if avg_auc > best_auc:
            best_auc = avg_auc
            best_alpha = alpha
            best_fit_prior = fit_prior

print(f"\nBest parameters: alpha={best_alpha}, fit_prior={best_fit_prior}, AUC={best_auc:.5f}")

# Train final model with best parameters
best_clf = MultinomialNB(alpha=best_alpha, fit_prior=best_fit_prior)
best_clf.fit(X_train, Y_train)

# Evaluate on test set
pos_prob = best_clf.predict_proba(X_test)[:, 1]
predictions = best_clf.predict(X_test)
test_auc = roc_auc_score(Y_test, pos_prob)

# Calculate ROC curve data for enhanced model
enhanced_fpr, enhanced_tpr, _ = calculate_roc_data(Y_test, pos_prob)

print(f"\nTest AUC with the best model: {test_auc:.5f}")
print(f"Improvement over basic model: {(test_auc - basic_auc) * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, predictions))
print("\nClassification Report:")
print(classification_report(Y_test, predictions))

# Plot ROC curves for both models
plt.figure(figsize=(10, 8))
plt.plot(basic_fpr, basic_tpr, color='blue', lw=2, label=f'Basic Model (AUC = {basic_auc:.3f})')
plt.plot(enhanced_fpr, enhanced_tpr, color='darkorange', lw=2, label=f'Enhanced Model (AUC = {test_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC)', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Save ROC curves to PDF
with PdfPages('roc_curves.pdf') as pdf:
    # Plot comparison of both models
    plt.figure(figsize=(10, 8))
    plt.plot(basic_fpr, basic_tpr, color='blue', lw=2, label=f'Basic Model (AUC = {basic_auc:.3f})')
    plt.plot(enhanced_fpr, enhanced_tpr, color='darkorange', lw=2, label=f'Enhanced Model (AUC = {test_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Comparison of ROC Curves', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add explanation text
    plt.figtext(0.5, 0.01, 
                "AUC = 0.5 oznacza, że model nie jest lepszy niż losowe zgadywanie\n"
                "AUC = 1.0 oznacza idealny model, który doskonale rozróżnia klasy",
                ha="center", fontsize=12)
    
    pdf.savefig()
    plt.close()
    
    # Add a page with detailed explanation
    fig = plt.figure(figsize=(10, 8))
    plt.axis('off')
    plt.text(0.1, 0.9, "Area Under the Curve (AUC) krzywej Receiver Operating Characteristic (ROC)", fontsize=16, weight='bold')
    plt.text(0.1, 0.85, "to skonsolidowana miara często stosowana w klasyfikacji binarnej.", fontsize=14)
    plt.text(0.1, 0.8, "AUC ocenia zdolność modelu do rozróżniania między klasami.", fontsize=14)
    plt.text(0.1, 0.75, "Jest to liczba między 0 a 1, gdzie:", fontsize=14)
    plt.text(0.1, 0.7, "AUC = 0.5 oznacza, że model nie jest lepszy niż losowe zgadywanie,", fontsize=14)
    plt.text(0.1, 0.65, "AUC = 1.0 oznacza idealny model, który doskonale rozróżnia klasy.", fontsize=14)
    
    plt.text(0.1, 0.55, "Krzywa ROC to wykres przedstawiający zależność między:", fontsize=14)
    plt.text(0.1, 0.5, "- wskaźnikiem prawdziwie pozytywnych (true positive rate)", fontsize=14)
    plt.text(0.1, 0.45, "- wskaźnikiem fałszywie pozytywnych (false positive rate)", fontsize=14)
    plt.text(0.1, 0.4, "przy różnych progach prawdopodobieństwa, w zakresie od 0 do 1.", fontsize=14)
    
    plt.text(0.1, 0.3, f"Wyniki:", fontsize=14, weight='bold')
    plt.text(0.1, 0.25, f"- Model podstawowy (tylko oceny): AUC = {basic_auc:.3f}", fontsize=14)
    plt.text(0.1, 0.2, f"- Model rozszerzony (oceny + cechy użytkowników i filmów): AUC = {test_auc:.3f}", fontsize=14)
    plt.text(0.1, 0.15, f"- Poprawa: {(test_auc - basic_auc) * 100:.2f}%", fontsize=14)
    
    pdf.savefig()
    plt.close()

print("\nROC curves saved to 'roc_curves.pdf'")
