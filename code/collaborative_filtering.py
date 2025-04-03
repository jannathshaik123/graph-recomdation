import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Paths to your data and models
base_path = os.path.join(os.path.dirname(os.getcwd()), "data/processed")
model_path = os.path.join(os.path.dirname(os.getcwd()), "models")

item_based_model_path = os.path.join(model_path,'item_based_cf_model.pkl')
svd_model_path = os.path.join(model_path,'svd_model.pkl')

print("Loading models and data...")

# Load the models
with open(item_based_model_path, 'rb') as f:
    model_knn = pickle.load(f)
    
with open(svd_model_path, 'rb') as f:
    svd_model = pickle.load(f)

# Load the user-item matrix
user_item_df = pd.read_csv(os.path.join(base_path, 'user_item_matrix.csv'), index_col=0)
user_item_df = user_item_df.fillna(0)

# Load or create validation data
# Option 1: If you have ground truth data to evaluate against:
try:
    # Try to load the full dataset that includes ratings
    merged_data = pd.read_csv(os.path.join(base_path, 'yelp_recommendation_data.csv'))
    print("Creating validation set from full dataset...")
    # Create a validation set (20% of data)
    _, val_data = train_test_split(merged_data, test_size=0.2, random_state=42)
except FileNotFoundError:
    # Option 2: If you don't have the full dataset, create a validation set from the matrix
    print("Full dataset not found. Creating validation set from user-item matrix...")
    # Get non-zero entries from the matrix to use as validation
    validation_pairs = []
    for user_id in user_item_df.index[:100]:  # Limit to first 100 users for efficiency
        for business_id in user_item_df.columns:
            rating = user_item_df.loc[user_id, business_id]
            if rating > 0:  # This is a real rating
                validation_pairs.append({
                    'user_id': user_id,
                    'business_id': business_id,
                    'stars': rating
                })
                if len(validation_pairs) >= 1000:  # Limit to 1000 pairs for efficiency
                    break
        if len(validation_pairs) >= 1000:
            break
    
    val_data = pd.DataFrame(validation_pairs)

print(f"Validation set contains {len(val_data)} ratings.")

# Define prediction functions (copied from your original script)
def user_based_cf(user_item_matrix, user_id, business_id, k=10):
    """User-based collaborative filtering to predict rating"""
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

def item_based_cf(user_item_matrix, user_id, business_id, model_knn, k=10):
    """Item-based collaborative filtering to predict rating"""
    # Check if user or business is not in the matrix
    if user_id not in user_item_matrix.index or business_id not in user_item_matrix.columns:
        return 3.5  # Return average rating if user or business is not in training data
    
    # Transpose the matrix for item-based similarity
    item_item_matrix = user_item_matrix.T
    
    # Get business index
    try:
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
    except:
        return 3.5  # Return average rating if any issues occur

def svd_predict(user_item_matrix, user_id, business_id, svd_model):
    """SVD-based prediction"""
    # Check if user or business is not in the matrix
    if user_id not in user_item_matrix.index or business_id not in user_item_matrix.columns:
        return 3.5  # Return average rating if user or business is not in training data
    
    try:
        # Get user and business indices
        user_idx = user_item_matrix.index.get_loc(user_id)
        business_idx = user_item_matrix.columns.get_loc(business_id)
        
        # Convert matrix to sparse format
        from scipy.sparse import csr_matrix
        user_item_sparse = csr_matrix(user_item_matrix.values)
        
        # Get the latent factors
        user_factors = svd_model.transform(user_item_sparse)[user_idx]
        business_factors = svd_model.components_[:, business_idx]
        
        # Calculate the predicted rating
        predicted_rating = np.dot(user_factors, business_factors)
        
        # Clip the rating to be between 1 and 5
        predicted_rating = max(1, min(5, predicted_rating))
        
        return predicted_rating
    except:
        return 3.5  # Return average rating if any issues occur

def evaluate_model(model_name, true_ratings, predicted_ratings):
    """Evaluate model performance using RMSE and MAE"""
    rmse = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
    mae = mean_absolute_error(true_ratings, predicted_ratings)
    
    print(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    return rmse, mae

# Function to make predictions with each model individually
def make_individual_predictions(val_data, user_item_df, model_knn, svd_model, sample_size=None):
    """Make predictions with each model individually and evaluate"""
    if sample_size and sample_size < len(val_data):
        val_sample = val_data.sample(sample_size, random_state=42)
    else:
        val_sample = val_data
    
    print(f"Making predictions on {len(val_sample)} validation samples...")
    
    # Prepare containers for predictions
    user_based_preds = []
    item_based_preds = []
    svd_preds = []
    
    # Process in smaller batches to show progress
    batch_size = 100
    for i in range(0, len(val_sample), batch_size):
        batch = val_sample.iloc[i:i+batch_size]
        
        batch_user_based = []
        batch_item_based = []
        batch_svd = []
        
        for _, row in batch.iterrows():
            user_id = row['user_id']
            business_id = row['business_id']
            
            # User-based CF
            ub_pred = user_based_cf(user_item_df, user_id, business_id)
            batch_user_based.append(ub_pred)
            
            # Item-based CF
            try:
                ib_pred = item_based_cf(user_item_df, user_id, business_id, model_knn)
                batch_item_based.append(ib_pred)
            except Exception as e:
                print(f"Error in item-based CF: {e}")
                batch_item_based.append(3.5)
            
            # SVD
            try:
                svd_pred = svd_predict(user_item_df, user_id, business_id, svd_model)
                batch_svd.append(svd_pred)
            except Exception as e:
                print(f"Error in SVD: {e}")
                batch_svd.append(3.5)
        
        # Append batch predictions
        user_based_preds.extend(batch_user_based)
        item_based_preds.extend(batch_item_based)
        svd_preds.extend(batch_svd)
        
        print(f"Processed {i+len(batch)}/{len(val_sample)} predictions")
    
    # Evaluate each model
    true_ratings = val_sample['stars'].values
    
    print("\nModel Evaluation Results:")
    evaluate_model("User-based CF", true_ratings, user_based_preds)
    evaluate_model("Item-based CF", true_ratings, item_based_preds)
    evaluate_model("SVD", true_ratings, svd_preds)
    
    # Try simple ensemble (average of all models)
    ensemble_preds = [(u + i + s) / 3 for u, i, s in zip(user_based_preds, item_based_preds, svd_preds)]
    evaluate_model("Ensemble (Simple Average)", true_ratings, ensemble_preds)
    
    # Try weighted ensemble
    weights = [0.4, 0.3, 0.3]  # Same as in your original script
    weighted_preds = [(u*weights[0] + i*weights[1] + s*weights[2]) 
                      for u, i, s in zip(user_based_preds, item_based_preds, svd_preds)]
    evaluate_model("Ensemble (Weighted Average)", true_ratings, weighted_preds)
    
    return {
        'user_based': user_based_preds,
        'item_based': item_based_preds,
        'svd': svd_preds,
        'ensemble': ensemble_preds,
        'weighted': weighted_preds
    }

# Run the evaluation
print("Starting model evaluation...")
# Limit to 1000 samples for faster execution
results = make_individual_predictions(val_data, user_item_df, model_knn, svd_model, sample_size=1000)

# Optionally, create a submission file with the best performing model
# (Check which model had the best RMSE and use that one)
print("\nYou can now create a submission file using the best performing model.")