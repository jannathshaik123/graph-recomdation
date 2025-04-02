import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Set the base path for data files
base_path = os.path.join(os.path.dirname(os.getcwd()), "data\processed")
print("Loading the user-item matrix...")
# Load the user-item matrix created in the preprocessing step
user_item_df = pd.read_csv(os.path.join(base_path,'user_item_matrix.csv'), index_col=0)

# Fill NaN values with 0
user_item_df = user_item_df.fillna(0)

# Create a sparse matrix for efficiency
user_item_sparse = csr_matrix(user_item_df.values)

# Load prediction pairs (test set)
prediction_pairs = pd.read_csv(os.path.join(base_path,'prediction_pairs.csv'))

print("Building collaborative filtering models...")

# 1. User-based collaborative filtering
def user_based_cf(user_item_matrix, user_id, business_id, k=10):
    """
    User-based collaborative filtering to predict rating
    
    Args:
        user_item_matrix: DataFrame of user-item matrix
        user_id: Target user ID
        business_id: Target business ID
        k: Number of similar users to consider
        
    Returns:
        Predicted rating
    """
    # Check if user or business is not in the matrix
    if user_id not in user_item_matrix.index or business_id not in user_item_matrix.columns:
        return 3.5  # Return average rating if user or business is not in training data
    
    # Get user's row
    user_row = user_item_matrix.loc[user_id]
    
    # Calculate similarities with other users
    similarities = []
    for other_user in user_item_matrix.index:
        if other_user == user_id:
            continue
        
        other_row = user_item_matrix.loc[other_user]
        
        # Find businesses both users have rated
        common_businesses = user_row.index[(user_row > 0) & (other_row > 0)]
        
        if len(common_businesses) == 0:
            continue
            
        # Calculate cosine similarity between the two users based on common businesses
        user_ratings = user_row[common_businesses].values
        other_ratings = other_row[common_businesses].values
        
        # Normalize ratings
        user_ratings = user_ratings - np.mean(user_ratings)
        other_ratings = other_ratings - np.mean(other_ratings)
        
        # Calculate similarity (avoid division by zero)
        norm_product = np.linalg.norm(user_ratings) * np.linalg.norm(other_ratings)
        if norm_product == 0:
            similarity = 0
        else:
            similarity = np.dot(user_ratings, other_ratings) / norm_product
            
        similarities.append((other_user, similarity, other_row[business_id]))
    
    # Sort by similarity and take top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k = similarities[:k]
    
    # If no similar users found or none rated the business
    if not top_k or all(rating == 0 for _, _, rating in top_k):
        return 3.5  # Return average rating
    
    # Calculate weighted average rating
    weighted_sum = sum(sim * rating for _, sim, rating in top_k if rating > 0)
    sum_similarities = sum(sim for _, sim, _ in top_k if sim > 0)
    
    if sum_similarities == 0:
        return 3.5  # Return average rating
    
    return weighted_sum / sum_similarities

# 2. Item-based collaborative filtering using KNN
print("Training item-based collaborative filtering model...")
# Transpose the matrix to get item-item similarity
item_item_matrix = user_item_df.T

# Create a KNN model for item-based CF
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(item_item_matrix.values)

# Save the model
with open('item_based_cf_model.pkl', 'wb') as f:
    pickle.dump(model_knn, f)

def item_based_cf(user_item_matrix, user_id, business_id, model_knn, k=10):
    """
    Item-based collaborative filtering to predict rating
    
    Args:
        user_item_matrix: DataFrame of user-item matrix
        user_id: Target user ID
        business_id: Target business ID
        model_knn: Trained KNN model
        k: Number of similar items to consider
        
    Returns:
        Predicted rating
    """
    # Check if user or business is not in the matrix
    if user_id not in user_item_matrix.index or business_id not in user_item_matrix.columns:
        return 3.5  # Return average rating if user or business is not in training data
    
    # Get business index
    business_idx = user_item_matrix.columns.get_loc(business_id)
    
    # Find k nearest neighbors
    distances, indices = model_knn.kneighbors(
        item_item_matrix.iloc[business_idx, :].values.reshape(1, -1), 
        n_neighbors=k+1
    )
    
    # Get similar businesses (excluding the business itself)
    similar_businesses = [(item_item_matrix.index[idx], distances[0][i]) 
                          for i, idx in enumerate(indices[0]) 
                          if item_item_matrix.index[idx] != business_id][:k]
    
    # Get user's ratings for similar businesses
    user_ratings = []
    for similar_business, distance in similar_businesses:
        if similar_business in user_item_matrix.columns:
            rating = user_item_matrix.loc[user_id, similar_business]
            if rating > 0:  # Only consider rated items
                # Convert distance to similarity (1 - distance)
                similarity = 1 - distance
                user_ratings.append((rating, similarity))
    
    # If user hasn't rated any similar businesses
    if not user_ratings:
        return 3.5  # Return average rating
    
    # Calculate weighted average rating
    weighted_sum = sum(rating * sim for rating, sim in user_ratings)
    sum_similarities = sum(sim for _, sim in user_ratings)
    
    if sum_similarities == 0:
        return 3.5  # Return average rating
    
    return weighted_sum / sum_similarities

# 3. Matrix Factorization (SVD)
print("Training SVD model...")
from sklearn.decomposition import TruncatedSVD

# Create SVD model
n_components = min(50, min(user_item_df.shape) - 1)  # Number of latent factors
svd = TruncatedSVD(n_components=n_components, random_state=42)
svd.fit(user_item_sparse)

# Save the model
with open('svd_model.pkl', 'wb') as f:
    pickle.dump(svd, f)

def svd_predict(user_item_matrix, user_id, business_id, svd_model):
    """
    SVD-based prediction
    
    Args:
        user_item_matrix: DataFrame of user-item matrix
        user_id: Target user ID
        business_id: Target business ID
        svd_model: Trained SVD model
        
    Returns:
        Predicted rating
    """
    # Check if user or business is not in the matrix
    if user_id not in user_item_matrix.index or business_id not in user_item_matrix.columns:
        return 3.5  # Return average rating if user or business is not in training data
    
    # Get user and business indices
    user_idx = user_item_matrix.index.get_loc(user_id)
    business_idx = user_item_matrix.columns.get_loc(business_id)
    
    # Get the latent factors
    user_factors = svd_model.transform(user_item_sparse)[user_idx]
    business_factors = svd_model.components_[:, business_idx]
    
    # Calculate the predicted rating
    predicted_rating = np.dot(user_factors, business_factors)
    
    # Clip the rating to be between 1 and 5
    predicted_rating = max(1, min(5, predicted_rating))
    
    return predicted_rating

# 4. Hybrid model (combining the predictions)
def hybrid_predict(user_item_matrix, user_id, business_id, model_knn, svd_model, weights=[0.4, 0.3, 0.3]):
    """
    Hybrid prediction combining user-based CF, item-based CF, and SVD
    
    Args:
        user_item_matrix: DataFrame of user-item matrix
        user_id: Target user ID
        business_id: Target business ID
        model_knn: Trained KNN model
        svd_model: Trained SVD model
        weights: Weights for each model [user_based, item_based, svd]
        
    Returns:
        Predicted rating
    """
    # Get predictions from each model
    user_based_pred = user_based_cf(user_item_matrix, user_id, business_id)
    item_based_pred = item_based_cf(user_item_matrix, user_id, business_id, model_knn)
    svd_pred = svd_predict(user_item_matrix, user_id, business_id, svd_model)
    
    # Combine predictions using weighted average
    hybrid_pred = (
        weights[0] * user_based_pred + 
        weights[1] * item_based_pred + 
        weights[2] * svd_pred
    )
    
    return hybrid_pred

# Make predictions for the test set
print("Making predictions for the test set...")
predictions = []

# Load the models
with open('item_based_cf_model.pkl', 'rb') as f:
    model_knn = pickle.load(f)
    
with open('svd_model.pkl', 'rb') as f:
    svd_model = pickle.load(f)

# Process in batches to avoid memory issues
batch_size = 1000
for i in range(0, len(prediction_pairs), batch_size):
    batch = prediction_pairs.iloc[i:i+batch_size]
    
    batch_predictions = []
    for _, row in batch.iterrows():
        user_id = row['user_id']
        business_id = row['business_id']
        
        # Use hybrid model for prediction
        pred = hybrid_predict(user_item_df, user_id, business_id, model_knn, svd_model)
        batch_predictions.append(pred)
    
    predictions.extend(batch_predictions)
    print(f"Processed {i+len(batch)}/{len(prediction_pairs)} predictions")

# Create submission file
submission = prediction_pairs.copy()
submission['stars'] = predictions
submission.to_csv('submission.csv', index=False)

print("Recommendation system modeling complete!")
print("Predictions saved to submission.csv")

# Optional: Evaluate on a validation set
# If you want to evaluate the model before submitting, you can split the training data
# into a training and validation set and evaluate the performance

def evaluate_model(model_name, true_ratings, predicted_ratings):
    """
    Evaluate model performance using RMSE and MAE
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    rmse = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
    mae = mean_absolute_error(true_ratings, predicted_ratings)
    
    print(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    return rmse, mae

# Example of creating a validation set (if needed)
from sklearn.model_selection import train_test_split

# Load the merged data
merged_data = pd.read_csv('yelp_recommendation_data.csv')

# Split into training and validation sets
train_data, val_data = train_test_split(merged_data, test_size=0.2, random_state=42)

# Create user-item matrices
train_matrix = train_data.pivot_table(index='user_id', columns='business_id', values='stars').fillna(0)

# Make predictions on validation set
val_predictions = []
for _, row in val_data.iterrows():
    user_id = row['user_id']
    business_id = row['business_id']
    pred = hybrid_predict(train_matrix, user_id, business_id, model_knn, svd_model)
    val_predictions.append(pred)

# Evaluate
evaluate_model("Hybrid Model", val_data['stars'].values, val_predictions)
