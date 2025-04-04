import pandas as pd
import numpy as np
import pickle
import os
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Memory-efficient collaborative filtering implementation
class MemoryEfficientCF:
    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors
        self.user_similarity = None
        self.item_similarity = None
        self.user_to_idx = None
        self.business_to_idx = None
        self.idx_to_user = None
        self.idx_to_business = None
    
    def fit(self, ratings_matrix):
        """
        Train the model using the provided ratings matrix
        
        Parameters:
        ratings_matrix (scipy.sparse.csr_matrix): User-item ratings matrix
        """
        print("Computing similarity matrices...")
        # For user-based CF, compute similarity between users
        self.user_similarity = cosine_similarity(ratings_matrix, dense_output=False)
        
        # For item-based CF, compute similarity between items
        self.item_similarity = cosine_similarity(ratings_matrix.T, dense_output=False)
        
        print("Similarity matrices computed successfully")
        return self
    
    def predict_user_based(self, ratings_matrix, user_id, item_id):
        """
        Predict rating for a specific user-item pair using user-based CF
        """
        if self.user_similarity is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Get similar users
        user_sim_scores = self.user_similarity[user_id].toarray().flatten()
        
        # Get ratings of all users for this item
        item_ratings = ratings_matrix[:, item_id].toarray().flatten()
        
        # Mask users who haven't rated this item
        mask = item_ratings > 0
        if not mask.any():
            return 0  # No user has rated this item
        
        # Get top similar users who rated this item
        similar_users = user_sim_scores[mask]
        item_ratings = item_ratings[mask]
        
        # Sort by similarity
        sorted_idx = np.argsort(similar_users)[::-1]
        top_n_idx = sorted_idx[:self.n_neighbors]
        
        # If no similar users, return average rating
        if len(top_n_idx) == 0:
            return np.mean(item_ratings) if len(item_ratings) > 0 else 0
        
        # Get ratings of top similar users
        top_sim_users = similar_users[top_n_idx]
        top_ratings = item_ratings[top_n_idx]
        
        # If all similarities are zero, return the mean rating
        if np.sum(top_sim_users) == 0:
            return np.mean(top_ratings)
        
        # Weighted average of ratings
        pred = np.sum(top_sim_users * top_ratings) / np.sum(top_sim_users)
        return pred
    
    def predict_item_based(self, ratings_matrix, user_id, item_id):
        """
        Predict rating for a specific user-item pair using item-based CF
        """
        if self.item_similarity is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Get user's ratings
        user_ratings = ratings_matrix[user_id].toarray().flatten()
        
        # Mask items rated by user
        mask = user_ratings > 0
        if not mask.any():
            return 0  # User has not rated any items
        
        # Get similarity scores between target item and items rated by user
        item_sim_scores = self.item_similarity[item_id, mask].toarray().flatten()
        user_ratings = user_ratings[mask]
        
        # Sort by similarity
        sorted_idx = np.argsort(item_sim_scores)[::-1]
        top_n_idx = sorted_idx[:self.n_neighbors]
        
        # If no similar items, return average rating
        if len(top_n_idx) == 0:
            return np.mean(user_ratings) if len(user_ratings) > 0 else 0
        
        # Get top similar items
        top_sim_items = item_sim_scores[top_n_idx]
        top_ratings = user_ratings[top_n_idx]
        
        # If all similarities are zero, return the mean rating
        if np.sum(top_sim_items) == 0:
            return np.mean(top_ratings)
        
        # Weighted average of ratings
        pred = np.sum(top_sim_items * top_ratings) / np.sum(top_sim_items)
        return pred
    
    def predict(self, ratings_matrix, user_id, item_id, method='hybrid'):
        """
        Predict rating for user-item pair using the specified method
        
        Parameters:
        method (str): 'user', 'item', or 'hybrid'
        """
        if method == 'user':
            return self.predict_user_based(ratings_matrix, user_id, item_id)
        elif method == 'item':
            return self.predict_item_based(ratings_matrix, user_id, item_id)
        elif method == 'hybrid':
            user_pred = self.predict_user_based(ratings_matrix, user_id, item_id)
            item_pred = self.predict_item_based(ratings_matrix, user_id, item_id)
            return (user_pred + item_pred) / 2
        else:
            raise ValueError("Method must be 'user', 'item', or 'hybrid'")

    def recommend_top_n(self, ratings_matrix, user_id, n=10, method='hybrid'):
        """
        Recommend top N items for a user
        """
        # Get all items user hasn't rated
        user_ratings = ratings_matrix[user_id].toarray().flatten()
        unrated_items = np.where(user_ratings == 0)[0]
        
        # Predict ratings for all unrated items
        predictions = []
        for item_id in unrated_items:
            pred = self.predict(ratings_matrix, user_id, item_id, method)
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
        
        # Save mappings and parameters
        model_params = {
            'n_neighbors': self.n_neighbors,
            'user_to_idx': self.user_to_idx,
            'business_to_idx': self.business_to_idx,
            'idx_to_user': self.idx_to_user,
            'idx_to_business': self.idx_to_business
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
        model = cls(n_neighbors=model_params['n_neighbors'])
        
        # Set mappings
        model.user_to_idx = model_params['user_to_idx']
        model.business_to_idx = model_params['business_to_idx']
        model.idx_to_user = model_params['idx_to_user']
        model.idx_to_business = model_params['idx_to_business']
        
        # Load similarity matrices if they exist
        if os.path.exists(os.path.join(folder_path, "user_similarity.npz")):
            model.user_similarity = load_npz(os.path.join(folder_path, "user_similarity.npz"))
        
        if os.path.exists(os.path.join(folder_path, "item_similarity.npz")):
            model.item_similarity = load_npz(os.path.join(folder_path, "item_similarity.npz"))
        
        print(f"Model loaded from {folder_path}")
        return model

# Load and prepare the data
def prepare_data(file_path, sample_size=None):
    """
    Load and prepare the data, with optional sampling for memory constraints
    """
    print("Loading data...")
    # Load data with lower memory usage
    dtypes = {
        'user_id': 'str', 
        'business_id': 'str',
        'stars': 'float32',
        'useful_votes': 'float32',
        'funny_votes': 'float32',
        'cool_votes': 'float32'
    }
    
    # Read data in chunks if dealing with large files
    df = pd.read_csv(file_path, dtype=dtypes)
    
    # Sample if needed
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
    
    # Extract only what we need for collaborative filtering
    ratings_df = df[['user_id', 'business_id', 'stars']]
    
    # Create mappings for user and business IDs to matrix indices
    user_ids = ratings_df['user_id'].unique()
    business_ids = ratings_df['business_id'].unique()
    
    user_to_idx = {user: idx for idx, user in enumerate(user_ids)}
    business_to_idx = {business: idx for idx, business in enumerate(business_ids)}
    
    # Map IDs to indices
    ratings_df['user_idx'] = ratings_df['user_id'].map(user_to_idx)
    ratings_df['business_idx'] = ratings_df['business_id'].map(business_to_idx)
    
    # Create sparse ratings matrix
    ratings = csr_matrix((ratings_df['stars'], 
                         (ratings_df['user_idx'], ratings_df['business_idx'])),
                         shape=(len(user_ids), len(business_ids)))
    
    return ratings, ratings_df, user_to_idx, business_to_idx

# Save ratings matrix
def save_ratings_matrix(ratings_matrix, file_path="model/ratings_matrix.npz"):
    """
    Save ratings matrix to disk
    """
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    save_npz(file_path, ratings_matrix)
    print(f"Ratings matrix saved to {file_path}")

# Load ratings matrix
def load_ratings_matrix(file_path="model/ratings_matrix.npz"):
    """
    Load ratings matrix from disk
    """
    ratings_matrix = load_npz(file_path)
    print(f"Ratings matrix loaded from {file_path}")
    return ratings_matrix

# Evaluate model
def evaluate_model(model, ratings_matrix, test_set, method='hybrid'):
    """
    Evaluate the model using various metrics
    """
    predictions = []
    actuals = []
    
    for _, row in test_set.iterrows():
        user_idx = row['user_idx']
        item_idx = row['business_idx']
        actual_rating = row['stars']
        
        # Skip if user or item not in training set
        if user_idx >= ratings_matrix.shape[0] or item_idx >= ratings_matrix.shape[1]:
            continue
        
        pred_rating = model.predict(ratings_matrix, user_idx, item_idx, method)
        predictions.append(pred_rating)
        actuals.append(actual_rating)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    
    # Calculate additional metrics
    mse = mean_squared_error(actuals, predictions)
    
    # Calculate correlation between predicted and actual ratings
    correlation = np.corrcoef(predictions, actuals)[0, 1]
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse,
        'Correlation': correlation
    }

# Example usage with model saving and loading
def main():
    # Replace with your actual file path
    file_path =  os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),"data/preprocessed_data/train_features.csv")
    model_dir = "model"
    
    # For memory constraints, sample the data
    sample_size = 50000  # Adjust based on your machine's memory
    
    # Check if model exists
    if os.path.exists(os.path.join(model_dir, "model_params.pkl")):
        print("Loading existing model...")
        model = MemoryEfficientCF.load_model(model_dir)
        ratings_matrix = load_ratings_matrix(os.path.join(model_dir, "ratings_matrix.npz"))
        
        # Load test data for evaluation
        _, ratings_df, _, _ = prepare_data(file_path, sample_size)
        _, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    else:
        print("Training new model...")
        # Prepare data
        ratings_matrix, ratings_df, user_to_idx, business_to_idx = prepare_data(file_path, sample_size)
        
        # Split into train and test sets
        train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
        
        # Create new training matrix
        train_matrix = csr_matrix((train_df['stars'], 
                                  (train_df['user_idx'], train_df['business_idx'])),
                                  shape=ratings_matrix.shape)
        
        # Initialize and train model
        model = MemoryEfficientCF(n_neighbors=20)
        model.fit(train_matrix)
        model.set_mappings(user_to_idx, business_to_idx)
        
        # Save model and ratings matrix
        model.save_model(model_dir)
        save_ratings_matrix(train_matrix, os.path.join(model_dir, "ratings_matrix.npz"))
        
        ratings_matrix = train_matrix
    
    # Evaluate model
    print("Evaluating user-based CF...")
    user_metrics = evaluate_model(model, ratings_matrix, test_df, 'user')
    print(f"User-based metrics: {user_metrics}")
    
    print("Evaluating item-based CF...")
    item_metrics = evaluate_model(model, ratings_matrix, test_df, 'item')
    print(f"Item-based metrics: {item_metrics}")
    
    print("Evaluating hybrid CF...")
    hybrid_metrics = evaluate_model(model, ratings_matrix, test_df, 'hybrid')
    print(f"Hybrid metrics: {hybrid_metrics}")
    
    # Example recommendation for a user
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

# Function to recommend for a specific user
def recommend_for_user(user_id, n=5, method='hybrid', model_dir='model'):
    """
    Generate recommendations for a specific user
    """
    # Load model and data
    model = MemoryEfficientCF.load_model(model_dir)
    ratings_matrix = load_ratings_matrix(os.path.join(model_dir, "ratings_matrix.npz"))
    
    # Check if user exists
    if user_id not in model.user_to_idx:
        print(f"User {user_id} not found in training data")
        return []
    
    user_idx = model.user_to_idx[user_id]
    recommended_items = model.recommend_top_n(ratings_matrix, user_idx, n=n, method=method)
    
    # Format results
    recommendations = []
    for item_idx, predicted_rating in recommended_items:
        business_id = model.idx_to_business[item_idx]
        recommendations.append({
            'business_id': business_id,
            'predicted_rating': predicted_rating
        })
    
    return recommendations

if __name__ == "__main__":
    main()
    
    # Example of how to use the recommend_for_user function
    # user_id = "rLtl8ZkDX5vH5nAx9C3q5Q"  # Replace with an actual user ID
    # recommendations = recommend_for_user(user_id, n=10)
    # print(f"\nRecommendations for user {user_id}:")
    # for i, rec in enumerate(recommendations, 1):
    #     print(f"{i}. Business: {rec['business_id']}, Predicted Rating: {rec['predicted_rating']:.2f}")
    

    