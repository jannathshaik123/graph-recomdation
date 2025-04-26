import os
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from collections import defaultdict

class ImprovedCollaborativeFiltering:
    """
    Memory-efficient collaborative filtering with bounded predictions
    and memory optimization techniques.
    """
    def __init__(self, n_neighbors=20, min_rating=1, max_rating=5):
        self.n_neighbors = n_neighbors
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.user_similarity = None
        self.item_similarity = None
        self.user_to_idx = None
        self.business_to_idx = None
        self.idx_to_user = None
        self.idx_to_business = None
        self.user_means = None
        self.item_means = None
        self.global_mean = None
    
    def fit(self, ratings_matrix):
        """
        Train the model using the provided ratings matrix with normalization.
        
        Parameters:
        ratings_matrix (scipy.sparse.csr_matrix): User-item ratings matrix
        """
        print("Computing global statistics...")
        # Calculate user means (average rating for each user)
        user_sums = ratings_matrix.sum(axis=1).A.flatten()
        user_counts = (ratings_matrix > 0).sum(axis=1).A.flatten()
        self.user_means = np.zeros_like(user_sums, dtype=float)
        nonzero_mask = user_counts > 0
        self.user_means[nonzero_mask] = user_sums[nonzero_mask] / user_counts[nonzero_mask]
        
        # Calculate item means (average rating for each item)
        item_sums = ratings_matrix.sum(axis=0).A.flatten()
        item_counts = (ratings_matrix > 0).sum(axis=0).A.flatten()
        self.item_means = np.zeros_like(item_sums, dtype=float)
        nonzero_mask = item_counts > 0
        self.item_means[nonzero_mask] = item_sums[nonzero_mask] / item_counts[nonzero_mask]
        
        # Calculate global mean
        total_ratings = ratings_matrix.data.sum()
        total_count = len(ratings_matrix.data)
        self.global_mean = total_ratings / total_count if total_count > 0 else 0
        
        print("Creating normalized matrix for similarity calculation...")
        # Create normalized matrix for better similarity calculation
        normalized_matrix = self._create_normalized_matrix(ratings_matrix)
        
        print("Computing similarity matrices...")
        # Compute similarity matrices using the normalized data
        self.user_similarity = self._compute_similarity(normalized_matrix)
        self.item_similarity = self._compute_similarity(normalized_matrix.T)
        
        print("Model training completed")
        return self
    
    def _create_normalized_matrix(self, ratings_matrix):
        """Create a normalized version of the ratings matrix for similarity computation"""
        # Copy the matrix to avoid modifying the original
        normalized = ratings_matrix.copy()
        
        # Use mean-centering approach
        rows, cols = normalized.nonzero()
        for i, j in zip(rows, cols):
            user_mean = self.user_means[i]
            normalized[i, j] = normalized[i, j] - user_mean
        
        return normalized
    
    def _compute_similarity(self, matrix, batch_size=1000):
        """
        Compute similarity matrix in batches to manage memory usage
        """
        n_rows = matrix.shape[0]
        similarity = csr_matrix((n_rows, n_rows))
        
        # Process in batches to reduce memory usage
        for i in range(0, n_rows, batch_size):
            end = min(i + batch_size, n_rows)
            batch = matrix[i:end]
            
            # Compute similarity for this batch against all items
            batch_sim = cosine_similarity(batch, matrix, dense_output=False)
            
            # Store the batch similarity
            similarity[i:end] = batch_sim
            
            print(f"Processed similarities for rows {i} to {end} of {n_rows}")
        
        return similarity
    
    def predict_user_based(self, ratings_matrix, user_id, item_id):
        """
        Predict rating using user-based collaborative filtering with bounded output
        """
        if self.user_similarity is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Get user's mean rating
        user_mean = self.user_means[user_id]
        
        # Fall back to global mean if user has no ratings
        if user_mean == 0:
            user_mean = self.global_mean
        
        # Get similar users
        user_sim_scores = self.user_similarity[user_id].toarray().flatten()
        
        # Get ratings of all users for this item
        item_ratings = ratings_matrix[:, item_id].toarray().flatten()
        
        # Mask users who haven't rated this item
        mask = item_ratings > 0
        if not mask.any():
            # No user has rated this item, return user's mean or item mean
            item_mean = self.item_means[item_id]
            return item_mean if item_mean > 0 else user_mean
        
        # Get ratings and similarity scores of users who rated this item
        sim_users = user_sim_scores[mask]
        ratings = item_ratings[mask]
        
        # Get user means for normalization
        user_means_masked = self.user_means[mask]
        
        # Normalize the ratings (remove user bias)
        normalized_ratings = np.zeros_like(ratings, dtype=float)
        nonzero_means = user_means_masked > 0
        normalized_ratings[nonzero_means] = ratings[nonzero_means] - user_means_masked[nonzero_means]
        
        # Get top N similar users
        if len(sim_users) > self.n_neighbors:
            top_n_idx = np.argsort(sim_users)[-self.n_neighbors:]
            sim_users = sim_users[top_n_idx]
            normalized_ratings = normalized_ratings[top_n_idx]
        
        # If all similarities are zero or negative, return user's mean rating
        if np.sum(np.maximum(0, sim_users)) == 0:
            return user_mean
        
        # Only use positive similarities for prediction
        positive_mask = sim_users > 0
        if not positive_mask.any():
            return user_mean
        
        sim_users = sim_users[positive_mask]
        normalized_ratings = normalized_ratings[positive_mask]
        
        # Calculate weighted average and add back user's mean
        weighted_sum = np.sum(sim_users * normalized_ratings)
        prediction = user_mean + weighted_sum / np.sum(sim_users)
        
        # Bound prediction to valid range
        return max(min(prediction, self.max_rating), self.min_rating)
    
    def predict_item_based(self, ratings_matrix, user_id, item_id):
        """
        Predict rating using item-based collaborative filtering with bounded output
        """
        if self.item_similarity is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Get item's mean rating
        item_mean = self.item_means[item_id]
        
        # Fall back to global mean if item has no ratings
        if item_mean == 0:
            item_mean = self.global_mean
        
        # Get user's ratings
        user_ratings = ratings_matrix[user_id].toarray().flatten()
        
        # Mask items rated by user
        mask = user_ratings > 0
        if not mask.any():
            # User has not rated any items, return item mean
            user_mean = self.user_means[user_id]
            return user_mean if user_mean > 0 else item_mean
        
        # Get similarity scores between target item and items rated by user
        item_sim_scores = self.item_similarity[item_id].toarray().flatten()[mask]
        rated_items = np.where(mask)[0]
        
        # User's ratings for those items
        ratings = user_ratings[mask]
        
        # Get item means for normalization
        item_means_masked = self.item_means[rated_items]
        
        # Normalize the ratings (remove item bias)
        normalized_ratings = np.zeros_like(ratings, dtype=float)
        nonzero_means = item_means_masked > 0
        normalized_ratings[nonzero_means] = ratings[nonzero_means] - item_means_masked[nonzero_means]
        
        # Get top N similar items
        if len(item_sim_scores) > self.n_neighbors:
            top_n_idx = np.argsort(item_sim_scores)[-self.n_neighbors:]
            item_sim_scores = item_sim_scores[top_n_idx]
            normalized_ratings = normalized_ratings[top_n_idx]
        
        # If all similarities are zero or negative, return item's mean rating
        if np.sum(np.maximum(0, item_sim_scores)) == 0:
            return item_mean
        
        # Only use positive similarities for prediction
        positive_mask = item_sim_scores > 0
        if not positive_mask.any():
            return item_mean
        
        item_sim_scores = item_sim_scores[positive_mask]
        normalized_ratings = normalized_ratings[positive_mask]
        
        # Calculate weighted average and add back item's mean
        weighted_sum = np.sum(item_sim_scores * normalized_ratings)
        prediction = item_mean + weighted_sum / np.sum(item_sim_scores)
        
        # Bound prediction to valid range
        return max(min(prediction, self.max_rating), self.min_rating)
    
    def predict(self, ratings_matrix, user_id, item_id, method='hybrid'):
        """
        Predict rating for user-item pair using specified method with proper bounds
        
        Parameters:
        method (str): 'user', 'item', 'hybrid', or 'baseline'
        """
        if method == 'user':
            return self.predict_user_based(ratings_matrix, user_id, item_id)
        elif method == 'item':
            return self.predict_item_based(ratings_matrix, user_id, item_id)
        elif method == 'hybrid':
            user_pred = self.predict_user_based(ratings_matrix, user_id, item_id)
            item_pred = self.predict_item_based(ratings_matrix, user_id, item_id)
            
            # Weighted average of both methods
            pred = (user_pred + item_pred) / 2
            return max(min(pred, self.max_rating), self.min_rating)
        elif method == 'baseline':
            # Simple baseline using mean ratings
            user_mean = self.user_means[user_id] if self.user_means[user_id] > 0 else self.global_mean
            item_mean = self.item_means[item_id] if self.item_means[item_id] > 0 else self.global_mean
            baseline = (user_mean + item_mean) / 2
            return max(min(baseline, self.max_rating), self.min_rating)
        else:
            raise ValueError("Method must be 'user', 'item', 'hybrid', or 'baseline'")

    def recommend_top_n(self, ratings_matrix, user_idx, n=10, method='hybrid', min_predicted_rating=3.5):
        """
        Recommend top N items for a user that meet minimum rating threshold
        """
        # Get all items user hasn't rated
        user_ratings = ratings_matrix[user_idx].toarray().flatten()
        unrated_items = np.where(user_ratings == 0)[0]
        
        # To improve efficiency, first get a baseline prediction for all items
        # This helps us filter out obvious low-rating items
        predictions = []
        
        # Use baseline method first to quickly filter candidates
        print(f"Getting baseline predictions for {len(unrated_items)} items...")
        baseline_predictions = []
        for item_id in unrated_items:
            pred = self.predict(ratings_matrix, user_idx, item_id, 'baseline')
            if pred >= min_predicted_rating:
                baseline_predictions.append((item_id, pred))
        
        # Sort baseline predictions and take top candidates for detailed prediction
        baseline_predictions.sort(key=lambda x: x[1], reverse=True)
        candidates = [item_id for item_id, _ in baseline_predictions[:min(len(baseline_predictions), n*3)]]
        
        print(f"Computing detailed predictions for {len(candidates)} candidate items...")
        # Get detailed predictions for candidates
        for item_id in candidates:
            pred = self.predict(ratings_matrix, user_idx, item_id, method)
            if pred >= min_predicted_rating:
                predictions.append((item_id, pred))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N
        return predictions[:n]
    
    def set_mappings(self, user_to_idx, business_to_idx):
        """
        Set the ID to index mappings
        """
        self.user_to_idx = user_to_idx
        self.business_to_idx = business_to_idx
        self.idx_to_user = {idx: user for user, idx in user_to_idx.items()}
        self.idx_to_business = {idx: business for business, idx in business_to_idx.items()}
    
    def save_model(self, folder_path="model"):
        """
        Save the model to disk
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Save similarity matrices
        if self.user_similarity is not None:
            save_npz(os.path.join(folder_path, "user_similarity.npz"), self.user_similarity)
        
        if self.item_similarity is not None:
            save_npz(os.path.join(folder_path, "item_similarity.npz"), self.item_similarity)
        
        # Save means and other parameters
        model_params = {
            'n_neighbors': self.n_neighbors,
            'min_rating': self.min_rating,
            'max_rating': self.max_rating,
            'user_to_idx': self.user_to_idx,
            'business_to_idx': self.business_to_idx,
            'idx_to_user': self.idx_to_user,
            'idx_to_business': self.idx_to_business,
            'user_means': self.user_means,
            'item_means': self.item_means,
            'global_mean': self.global_mean
        }
        
        with open(os.path.join(folder_path, "model_params.pkl"), 'wb') as f:
            pickle.dump(model_params, f)
        
        print(f"Model saved to {folder_path}")
    
    @classmethod
    def load_model(cls, folder_path="model"):
        """
        Load the model from disk
        """
        # Load parameters
        with open(os.path.join(folder_path, "model_params.pkl"), 'rb') as f:
            model_params = pickle.load(f)
        
        # Create model instance
        model = cls(
            n_neighbors=model_params['n_neighbors'],
            min_rating=model_params['min_rating'],
            max_rating=model_params['max_rating']
        )
        
        # Set mappings and means
        model.user_to_idx = model_params['user_to_idx']
        model.business_to_idx = model_params['business_to_idx']
        model.idx_to_user = model_params['idx_to_user']
        model.idx_to_business = model_params['idx_to_business']
        model.user_means = model_params['user_means']
        model.item_means = model_params['item_means']
        model.global_mean = model_params['global_mean']
        
        # Load similarity matrices if they exist
        if os.path.exists(os.path.join(folder_path, "user_similarity.npz")):
            model.user_similarity = load_npz(os.path.join(folder_path, "user_similarity.npz"))
        
        if os.path.exists(os.path.join(folder_path, "item_similarity.npz")):
            model.item_similarity = load_npz(os.path.join(folder_path, "item_similarity.npz"))
        
        print(f"Model loaded from {folder_path}")
        return model

# Prepare the data with filtering to avoid sparsity issues
def prepare_data(file_path, sample_size=None, min_user_ratings=5, min_business_ratings=5):
    """
    Load and prepare the data with filtering for users and businesses with minimum ratings
    """
    print("Loading data...")
    # Load data with lower memory usage
    dtypes = {
        'user_id': 'str', 
        'business_id': 'str',
        'stars': 'float32'
    }
    
    # Read data
    df = pd.read_csv(file_path, dtype=dtypes)
    
    if sample_size is not None and sample_size < len(df):
        print(f"Sampling {sample_size} records from {len(df)} total records")
        df = df.sample(n=sample_size, random_state=42)
    
    print("Filtering users and businesses with minimum ratings...")
    # Filter users and businesses with minimum number of ratings
    user_counts = df['user_id'].value_counts()
    business_counts = df['business_id'].value_counts()
    
    active_users = user_counts[user_counts >= min_user_ratings].index
    popular_businesses = business_counts[business_counts >= min_business_ratings].index
    
    # Filter dataset to include only active users and popular businesses
    filtered_df = df[df['user_id'].isin(active_users) & df['business_id'].isin(popular_businesses)]
    
    print(f"Original data size: {len(df)}, Filtered data size: {len(filtered_df)}")
    print(f"Users: {len(active_users)}, Businesses: {len(popular_businesses)}")
    
    # Create mappings
    user_ids = filtered_df['user_id'].unique()
    business_ids = filtered_df['business_id'].unique()
    
    user_to_idx = {user: idx for idx, user in enumerate(user_ids)}
    business_to_idx = {business: idx for idx, business in enumerate(business_ids)}
    
    # Map IDs to indices
    filtered_df['user_idx'] = filtered_df['user_id'].map(user_to_idx)
    filtered_df['business_idx'] = filtered_df['business_id'].map(business_to_idx)
    
    # Create sparse ratings matrix
    ratings = csr_matrix(
        (filtered_df['stars'], (filtered_df['user_idx'], filtered_df['business_idx'])),
        shape=(len(user_ids), len(business_ids))
    )
    
    return ratings, filtered_df, user_to_idx, business_to_idx

# Evaluate model with comprehensive metrics
def evaluate_model(model, ratings_matrix, test_set, method='hybrid'):
    """
    Evaluate the model using various metrics with detailed analysis
    """
    print(f"Evaluating model using '{method}' method...")
    predictions = []
    actuals = []
    
    # Used for calculating coverage
    total_possible_pairs = len(test_set)
    predicted_pairs = 0
    
    # Track predictions by rating value for distribution analysis
    prediction_distribution = defaultdict(int)
    actual_distribution = defaultdict(int)
    
    start_time = time.time()
    
    for idx, row in enumerate(test_set.itertuples()):
        user_idx = row.user_idx
        item_idx = row.business_idx
        actual_rating = row.stars
        
        # Skip if user or item not in training set
        if user_idx >= ratings_matrix.shape[0] or item_idx >= ratings_matrix.shape[1]:
            continue
        
        # Get prediction
        try:
            pred_rating = model.predict(ratings_matrix, user_idx, item_idx, method)
            predictions.append(pred_rating)
            actuals.append(actual_rating)
            predicted_pairs += 1
            
            # Track distributions
            # Round predictions to nearest 0.5 for distribution analysis
            rounded_pred = round(pred_rating * 2) / 2
            prediction_distribution[rounded_pred] += 1
            actual_distribution[actual_rating] += 1
            
            # Print progress
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(test_set)} test instances...")
        except Exception as e:
            print(f"Error predicting for user {user_idx}, item {item_idx}: {e}")
    
    # Calculate metrics
    if not predictions:
        print("No valid predictions were made!")
        return {}
    
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    
    # Calculate correlation
    correlation = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
    
    # Calculate coverage
    coverage = predicted_pairs / total_possible_pairs if total_possible_pairs > 0 else 0
    
    # Calculate precision at different thresholds (for good recommendations)
    precision_at_4 = sum(1 for a, p in zip(actuals, predictions) if a >= 4 and p >= 4) / sum(1 for p in predictions if p >= 4) if sum(1 for p in predictions if p >= 4) > 0 else 0
    
    # Calculate recall at different thresholds
    recall_at_4 = sum(1 for a, p in zip(actuals, predictions) if a >= 4 and p >= 4) / sum(1 for a in actuals if a >= 4) if sum(1 for a in actuals if a >= 4) > 0 else 0
    
    # F1 score
    f1_at_4 = 2 * (precision_at_4 * recall_at_4) / (precision_at_4 + recall_at_4) if (precision_at_4 + recall_at_4) > 0 else 0
    
    # Time taken
    time_taken = time.time() - start_time
    
    # Plot distributions if there are enough predictions
    if len(predictions) > 10:
        plt.figure(figsize=(12, 6))
        
        # Plot prediction distribution
        plt.subplot(1, 2, 1)
        pred_values = sorted(prediction_distribution.keys())
        pred_counts = [prediction_distribution[v] for v in pred_values]
        plt.bar(pred_values, pred_counts)
        plt.title('Prediction Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        
        # Plot actual distribution
        plt.subplot(1, 2, 2)
        actual_values = sorted(actual_distribution.keys())
        actual_counts = [actual_distribution[v] for v in actual_values]
        plt.bar(actual_values, actual_counts)
        plt.title('Actual Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(f"rating_distributions_{method}.png")
        plt.close()
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse,
        'Correlation': correlation,
        'Coverage': coverage,
        'Precision@4': precision_at_4,
        'Recall@4': recall_at_4,
        'F1@4': f1_at_4,
        'Time_taken': time_taken,
        'Number_of_predictions': len(predictions)
    }

def main():
    """Main function to run the collaborative filtering system"""
    # Replace with your actual file path
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data/preprocessed_data/train_features.csv")
    model_dir = "model_new"  # Change to a new directory if needed
    
    # For memory constraints, sample the data
    sample_size = 100000  # Adjust based on machine's memory
    
    # Define minimum ratings for users and businesses to filter out sparse data
    min_user_ratings = 5
    min_business_ratings = 5
    
    # Check if model exists
    if os.path.exists(os.path.join(model_dir, "model_params.pkl")):
        print("Loading existing model...")
        model = ImprovedCollaborativeFiltering.load_model(model_dir)
        ratings_matrix = load_npz(os.path.join(model_dir, "ratings_matrix.npz"))
        
        # Load test data for evaluation
        _, all_data, _, _ = prepare_data(file_path, sample_size, min_user_ratings, min_business_ratings)
        train_idx, test_idx = train_test_split(np.arange(len(all_data)), test_size=0.2, random_state=42)
        test_df = all_data.iloc[test_idx]
    else:
        print("Training new model...")
        # Prepare data
        ratings_matrix, all_data, user_to_idx, business_to_idx = prepare_data(
            file_path, sample_size, min_user_ratings, min_business_ratings
        )
        
        # Split into train and test sets
        train_idx, test_idx = train_test_split(np.arange(len(all_data)), test_size=0.2, random_state=42)
        train_df = all_data.iloc[train_idx]
        test_df = all_data.iloc[test_idx]
        
        # Create training matrix
        train_matrix = csr_matrix(
            (train_df['stars'], (train_df['user_idx'], train_df['business_idx'])),
            shape=ratings_matrix.shape
        )
        
        # Initialize and train model
        model = ImprovedCollaborativeFiltering(n_neighbors=25, min_rating=1, max_rating=5)
        model.fit(train_matrix)
        model.set_mappings(user_to_idx, business_to_idx)
        
        # Save model and ratings matrix
        model.save_model(model_dir)
        save_npz(os.path.join(model_dir, "ratings_matrix.npz"), train_matrix)
        
        ratings_matrix = train_matrix
    
    # Evaluate model with different methods
    methods = ['user', 'item', 'hybrid', 'baseline']
    results = {}
    
    for method in methods:
        results[method] = evaluate_model(model, ratings_matrix, test_df, method)
        print(f"\n{method.upper()} Method Results:")
        for metric, value in results[method].items():
            print(f"{metric}: {value}")
    
    # Compare methods
    print("\nComparison of Methods:")
    metrics = ['RMSE', 'MAE', 'Precision@4', 'Recall@4', 'F1@4']
    for metric in metrics:
        print(f"\n{metric}:")
        for method in methods:
            print(f"  {method}: {results[method].get(metric, 'N/A')}")
    
    # Generate example recommendations
    if model.user_to_idx is not None:
        # Get a sample user ID
        sample_user_id = list(model.user_to_idx.keys())[0]
        user_idx = model.user_to_idx[sample_user_id]
        
        print(f"\nGenerating recommendations for user: {sample_user_id}")
        recommended_items = model.recommend_top_n(ratings_matrix, user_idx, n=5, method='hybrid')
        
        print("\nTop recommendations:")
        for item_idx, predicted_rating in recommended_items:
            business_id = model.idx_to_business[item_idx]
            print(f"Business ID: {business_id}, Predicted Rating: {predicted_rating:.2f}")

def recommend_for_user(user_id, n=5, method='hybrid', model_dir='model'):
    """
    Generate recommendations for a specific user with business details
    """
    # Load model and data
    model = ImprovedCollaborativeFiltering.load_model(model_dir)
    ratings_matrix = load_npz(os.path.join(model_dir, "ratings_matrix.npz"))
    
    # Check if user exists
    if user_id not in model.user_to_idx:
        print(f"User {user_id} not found in training data")
        return []
    
    # Load business data for extra information
    business_data_path = 'extract/business_data.csv'
    if os.path.exists(business_data_path):
        business_df = pd.read_csv(business_data_path)
        business_info = {row['business_id']: row for _, row in business_df.iterrows()}
    else:
        business_info = {}
    
    # Get recommendations
    user_idx = model.user_to_idx[user_id]
    recommended_items = model.recommend_top_n(ratings_matrix, user_idx, n=n, method=method)
    
    # Format results with business details if available
    recommendations = []
    for item_idx, predicted_rating in recommended_items:
        business_id = model.idx_to_business[item_idx]
        recommendation = {
            'business_id': business_id,
            'predicted_rating': predicted_rating
        }
        
        # Add business details if available
        if business_id in business_info:
            business = business_info[business_id]
            recommendation.update({
                'name': business.get('name', 'Unknown'),
                'city': business.get('city', 'Unknown'),
                'state': business.get('state', 'Unknown'),
                'categories': business.get('categories', '[]')
            })
        
        recommendations.append(recommendation)
    
    return recommendations

if __name__ == "__main__":
    main()