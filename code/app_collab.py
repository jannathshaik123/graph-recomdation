import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the necessary data and models

global user_item_df, business_profile, user_profile, model_knn, svd_model

@st.cache_resource
def load_data():
    base_path_1 = os.path.join(os.path.dirname(os.getcwd()), "data\processed")
    print(f"Loading data from {base_path_1}")
    
    user_item_df = pd.read_csv(os.path.join(base_path_1,'user_item_matrix.csv'), index_col=0)
    user_item_df = user_item_df.fillna(0)
    
    business_profile = pd.read_csv(os.path.join(base_path_1,'business_profile.csv'))
    
    user_profile = pd.read_csv(os.path.join(base_path_1,'user_profile.csv'))
    
    base_path_2 = os.path.join(os.path.dirname(os.getcwd()), "models")
    
    # Load models
    with open(os.path.join(base_path_2,'item_based_cf_model.pkl'), 'rb') as f:
        model_knn = pickle.load(f)
        
    with open(os.path.join(base_path_2,'svd_model.pkl'), 'rb') as f:
        svd_model = pickle.load(f)
    
    return user_item_df, business_profile, user_profile, model_knn, svd_model

# Define the prediction functions
def user_based_cf(user_item_matrix, user_id, business_id, k=10):
    """User-based collaborative filtering"""
    # Implementation as in the previous code
    # ...
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

def item_based_cf(user_item_matrix, user_id, business_id, model_knn, k=10):
    """Item-based collaborative filtering"""
    # Implementation as in the previous code
    # ...
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
    
    item_item_matrix = user_item_matrix.T
    
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

def svd_predict(user_item_matrix, user_id, business_id, svd_model):
    """SVD-based prediction"""
    # Implementation as in the previous code
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
    
    # Create a sparse matrix for efficiency
    user_item_sparse = csr_matrix(user_item_matrix.values)
    
    # Get the latent factors
    user_factors = svd_model.transform(user_item_sparse)[user_idx]
    business_factors = svd_model.components_[:, business_idx]
    
    # Calculate the predicted rating
    predicted_rating = np.dot(user_factors, business_factors)
    
    # Clip the rating to be between 1 and 5
    predicted_rating = max(1, min(5, predicted_rating))
    
    return predicted_rating

def hybrid_predict(user_item_matrix, user_id, business_id, model_knn, svd_model, weights=[0.4, 0.3, 0.3]):
    """Hybrid prediction combining multiple models"""
    # Implementation as in the previous code
    # ...
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
    # Normalize to a scale of 1-5
    return hybrid_pred

# Streamlit app
def main():
    st.title("Yelp Recommendation System")
    st.write("This app demonstrates a hybrid recommendation system built on Yelp data from Phoenix, AZ.")
    
    # Load data
    with st.spinner("Loading data and models..."):
        user_item_df, business_profile, user_profile, model_knn, svd_model = load_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "User Recommendations", "Business Analysis", "Model Evaluation"])
    
    if page == "Home":
        st.header("Welcome to the Yelp Recommendation System")
        st.write("""
        This application showcases a recommendation system built on Yelp data from Phoenix, AZ.
        
        The system uses multiple recommendation approaches:
        - User-based collaborative filtering
        - Item-based collaborative filtering
        - Matrix factorization (SVD)
        - A hybrid approach combining all three
        
        Use the sidebar to navigate to different sections of the app.
        """)
        
        # Display some stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Businesses", f"{len(business_profile):,}")
        with col2:
            st.metric("Users", f"{len(user_profile):,}")
        with col3:
            st.metric("Ratings", f"{user_item_df.notna().sum().sum():,}")
        
        # Show a sample of businesses
        st.subheader("Sample Businesses")
        st.dataframe(business_profile.sample(10)[['business_id', 'name', 'city', 'categories_str']])
    
    elif page == "User Recommendations":
        st.header("Get Personalized Recommendations")
        
        # User selection
        user_list = user_profile['user_id'].tolist()
        selected_user = st.selectbox("Select a user", user_list)
        
        if st.button("Get Recommendations"):
            with st.spinner("Generating recommendations..."):
                # Get user's rated businesses
                if selected_user in user_item_df.index:
                    user_ratings = user_item_df.loc[selected_user]
                    rated_businesses = user_ratings[user_ratings > 0].index.tolist()
                    
                    # Get businesses not rated by the user
                    unrated_businesses = [b for b in business_profile['business_id'].tolist() 
                                         if b not in rated_businesses]
                    
                    # Predict ratings for unrated businesses
                    predictions = []
                    for business_id in unrated_businesses[:100]:  # Limit to 100 for performance
                        pred = hybrid_predict(user_item_df, selected_user, business_id, model_knn, svd_model)
                        predictions.append((business_id, pred))
                    
                    # Sort by predicted rating
                    predictions.sort(key=lambda x: x[1], reverse=True)
                    
                    # Display top 10 recommendations
                    st.subheader("Top 10 Recommended Businesses")
                    
                    for i, (business_id, pred) in enumerate(predictions[:10]):
                        business_info = business_profile[business_profile['business_id'] == business_id].iloc[0]
                        
                        st.write(f"**{i+1}. {business_info['name']}** - Predicted Rating: {pred:.2f}⭐")
                        st.write(f"**Categories:** {business_info['categories_str']}")
                        st.write(f"**Location:** {business_info['city']}, {business_info['state']}")
                        st.write("---")
                else:
                    st.error("User has not rated any businesses in the training data.")
    
    elif page == "Business Analysis":
        st.header("Business Analysis")
        
        # Business selection
        business_list = business_profile['name'].tolist()
        selected_business_name = st.selectbox("Select a business", business_list)
        
        selected_business = business_profile[business_profile['name'] == selected_business_name].iloc[0]
        
        # Display business info
        st.subheader(f"{selected_business['name']}")
        st.write(f"**Categories:** {selected_business['categories_str']}")
        st.write(f"**Location:** {selected_business['city']}, {selected_business['state']}")
        
        if 'stars' in selected_business:
            st.write(f"**Average Rating:** {selected_business['stars']:.2f}⭐")
        
        st.write(f"**Review Count:** {selected_business['review_count']}")
        
        # Show similar businesses
        st.subheader("Similar Businesses")
        
        business_id = selected_business['business_id']
        if business_id in user_item_df.columns:
            # Get business index
            business_idx = user_item_df.columns.get_loc(business_id)
            
            # Find similar businesses using KNN
            business_matrix = user_item_df.T
            distances, indices = model_knn.kneighbors(
                business_matrix.iloc[business_idx, :].values.reshape(1, -1), 
                n_neighbors=6
            )
            
            # Display similar businesses (excluding itself)
            similar_businesses = []
            for i in range(1, len(indices[0])):
                idx = indices[0][i]
                similar_id = business_matrix.index[idx]
                similar_info = business_profile[business_profile['business_id'] == similar_id]
                
                if not similar_info.empty:
                    similar_businesses.append(similar_info.iloc[0])
            
            for i, business in enumerate(similar_businesses):
                st.write(f"**{i+1}. {business['name']}**")
                st.write(f"**Categories:** {business['categories_str']}")
                st.write(f"**Location:** {business['city']}, {business['state']}")
                st.write("---")
        else:
            st.warning("This business doesn't have enough ratings for similarity analysis.")
    
    elif page == "Model Evaluation":
        st.header("Model Evaluation")
        
        # Display RMSE comparison
        st.subheader("Model Performance Comparison")
        
        # Sample RMSE values (replace with actual values from your evaluation)
        models = ['User-based CF', 'Item-based CF', 'SVD', 'Hybrid']
        rmse_values = [1.05, 0.98, 0.92, 0.89]  # Example values
        
        # Create a bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(models, rmse_values, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        ax.set_ylabel('RMSE (lower is better)')
        ax.set_title('Recommendation Model Performance')
        ax.set_ylim(0, max(rmse_values) + 0.2)
        
        st.pyplot(fig)
        
        st.write("""
        **Root Mean Square Error (RMSE)** measures the square root of the average squared difference 
        between predicted and actual ratings. Lower values indicate better performance.
        
        The hybrid model combines the strengths of all three approaches to achieve the best performance.
        """)
        
        # Distribution of ratings
        st.subheader("Distribution of Ratings")
        
        # Get all non-zero ratings
        all_ratings = user_item_df.values.flatten()
        all_ratings = all_ratings[all_ratings > 0]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(all_ratings, bins=5, kde=True, ax=ax)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_title('Distribution of Ratings')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        
        st.pyplot(fig)

if __name__ == "__main__":
    main()
