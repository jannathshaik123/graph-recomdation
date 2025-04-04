import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
from io import BytesIO

# Import the collaborative filtering model
from utils import MemoryEfficientCF, load_ratings_matrix, recommend_for_user

# Set page config
st.set_page_config(
    page_title="Yelp Recommender System",
    page_icon="🍽️",
    layout="wide"
)

# Define paths
MODEL_DIR = os.path.join(os.path.dirname(os.getcwd()),"collabfiltering\model")
BUSINESS_DATA_PATH = r'extract\business_data.csv'  # We'll need to create this
USER_DATA_PATH = r'extract\user_data.csv'          # We'll need to create this

# Load model and matrices
@st.cache_resource
def load_model():
    try:
        model = MemoryEfficientCF.load_model(MODEL_DIR)
        ratings_matrix = load_ratings_matrix(os.path.join(MODEL_DIR, "ratings_matrix.npz"))
        return model, ratings_matrix
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load business and user data
@st.cache_data
def load_business_data():
    if os.path.exists(BUSINESS_DATA_PATH):
        return pd.read_csv(BUSINESS_DATA_PATH)
    else:
        st.warning(f"Business data file not found at {BUSINESS_DATA_PATH}. Only business IDs will be displayed.")
        return None

@st.cache_data
def load_user_data():
    if os.path.exists(USER_DATA_PATH):
        return pd.read_csv(USER_DATA_PATH)
    else:
        st.warning(f"User data file not found at {USER_DATA_PATH}. Only user IDs will be displayed.")
        return None

# Function to get business details
def get_business_details(business_id, business_df):
    if business_df is not None and business_id in business_df['business_id'].values:
        business = business_df[business_df['business_id'] == business_id].iloc[0]
        return {
            'name': business.get('name', 'Unknown'),
            'address': business.get('full_address', 'Unknown'),
            'city': business.get('city', 'Unknown'),
            'state': business.get('state', 'Unknown'),
            'stars': business.get('stars', 'Unknown'),
            'categories': business.get('categories', 'Unknown')
        }
    return {
        'name': 'Unknown',
        'address': 'Unknown',
        'city': 'Unknown',
        'state': 'Unknown',
        'stars': 'Unknown',
        'categories': 'Unknown'
    }

# Function to get user details
def get_user_details(user_id, user_df):
    if user_df is not None and user_id in user_df['user_id'].values:
        user = user_df[user_df['user_id'] == user_id].iloc[0]
        return {
            'name': user.get('name', 'Unknown'),
            'review_count': user.get('review_count', 'Unknown'),
            'average_stars': user.get('average_stars', 'Unknown')
        }
    return {
        'name': 'Unknown',
        'review_count': 'Unknown',
        'average_stars': 'Unknown'
    }

# Function to generate recommendations
def generate_recommendations(model, ratings_matrix, user_id, n=5, method='hybrid'):
    if user_id not in model.user_to_idx:
        return None, "User not found in the dataset"
    
    user_idx = model.user_to_idx[user_id]
    recommended_items = model.recommend_top_n(ratings_matrix, user_idx, n=n, method=method)
    
    recommendations = []
    for item_idx, predicted_rating in recommended_items:
        business_id = model.idx_to_business[item_idx]
        recommendations.append({
            'business_id': business_id,
            'predicted_rating': predicted_rating
        })
    
    return recommendations, None

# Function to visualize user ratings
def visualize_user_ratings(user_id, model, ratings_matrix, business_df):
    if user_id not in model.user_to_idx:
        return None
    
    user_idx = model.user_to_idx[user_id]
    user_ratings = ratings_matrix[user_idx].toarray().flatten()
    rated_items = np.where(user_ratings > 0)[0]
    
    if len(rated_items) == 0:
        return None
    
    ratings = []
    names = []
    
    for item_idx in rated_items:
        business_id = model.idx_to_business[item_idx]
        rating = user_ratings[item_idx]
        
        business_details = get_business_details(business_id, business_df)
        business_name = business_details['name']
        
        ratings.append(rating)
        names.append(business_name if business_name != 'Unknown' else business_id[:10] + '...')
    
    # Sort by rating
    sorted_indices = np.argsort(ratings)[::-1]
    ratings = [ratings[i] for i in sorted_indices]
    names = [names[i] for i in sorted_indices]
    
    # Limit to top 10 for visibility
    if len(ratings) > 10:
        ratings = ratings[:10]
        names = names[:10]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, ratings, color=sns.color_palette("viridis", len(ratings)))
    
    ax.set_xlabel('Rating')
    ax.set_title(f'Top Ratings by User')
    ax.set_xlim(0, 5.5)
    
    # Add rating values on bars
    for i, v in enumerate(ratings):
        ax.text(v + 0.1, i, f"{v}", va='center')
    
    plt.tight_layout()
    return fig

# Main app
def main():
    st.title("Yelp Recommender System")
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Load data and model
    model, ratings_matrix = load_model()
    business_df = load_business_data()
    user_df = load_user_data()
    
    if model is None or ratings_matrix is None:
        st.error("Failed to load model. Please check the model directory.")
        return
    
    # User selection
    st.sidebar.header("Select User")
    
    # Get list of users
    user_ids = list(model.user_to_idx.keys())
    
    # Option to search for user
    search_method = st.sidebar.radio("User Selection Method", ["Browse Users", "Search by ID"])
    
    if search_method == "Browse Users":
        # Pagination for users
        users_per_page = 15
        total_pages = (len(user_ids) + users_per_page - 1) // users_per_page
        page = st.sidebar.number_input("Page", min_value=1, max_value=total_pages, value=1)
        
        start_idx = (page - 1) * users_per_page
        end_idx = min(start_idx + users_per_page, len(user_ids))
        
        displayed_users = user_ids[start_idx:end_idx]
        
        # Display user names if available
        user_options = []
        for user_id in displayed_users:
            user_details = get_user_details(user_id, user_df)
            if user_details['name'] != 'Unknown':
                user_options.append(f"{user_details['name']} ({user_id})")
            else:
                user_options.append(user_id)
        
        selected_user_option = st.sidebar.selectbox("Select User", user_options)
        
        # Extract user_id from selected option
        if "(" in selected_user_option and ")" in selected_user_option:
            selected_user_id = selected_user_option.split("(")[1].split(")")[0]
        else:
            selected_user_id = selected_user_option
    else:
        # Search by ID
        search_query = st.sidebar.text_input("Enter User ID")
        
        # Filter users based on search
        if search_query:
            matching_users = [user_id for user_id in user_ids if search_query.lower() in user_id.lower()]
            
            if not matching_users:
                st.sidebar.warning("No matching users found.")
                return
            
            # Display user names if available
            user_options = []
            for user_id in matching_users[:20]:  # Limit to 20 results
                user_details = get_user_details(user_id, user_df)
                if user_details['name'] != 'Unknown':
                    user_options.append(f"{user_details['name']} ({user_id})")
                else:
                    user_options.append(user_id)
            
            selected_user_option = st.sidebar.selectbox("Matching Users", user_options)
            
            # Extract user_id from selected option
            if "(" in selected_user_option and ")" in selected_user_option:
                selected_user_id = selected_user_option.split("(")[1].split(")")[0]
            else:
                selected_user_id = selected_user_option
        else:
            st.sidebar.info("Enter a search term to find users.")
            return
    
    # Recommendation settings
    st.sidebar.header("Recommendation Settings")
    num_recommendations = st.sidebar.slider("Number of Recommendations", 1, 20, 5)
    method = st.sidebar.radio("Recommendation Method", ["hybrid", "user", "item"], format_func=lambda x: x.capitalize() + " Filtering")
    
    # Generate recommendations button
    if st.sidebar.button("Generate Recommendations"):
        # Display user information
        st.header("User Information")
        user_details = get_user_details(selected_user_id, user_df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("User Name", user_details['name'])
        with col2:
            st.metric("User ID", selected_user_id)
        with col3:
            st.metric("Average Stars", user_details['average_stars'] if user_details['average_stars'] != 'Unknown' else "N/A")
        
        # Generate recommendations
        recommendations, error_message = generate_recommendations(
            model, ratings_matrix, selected_user_id, n=num_recommendations, method=method
        )
        
        if error_message:
            st.error(error_message)
            return
        
        # Show recommendations
        st.header(f"Top {num_recommendations} Recommendations")
        
        # Visualization of user ratings
        st.subheader("User Rating History")
        rating_chart = visualize_user_ratings(selected_user_id, model, ratings_matrix, business_df)
        
        if rating_chart:
            st.pyplot(rating_chart)
        else:
            st.info("No rating history available for this user.")
        
        # Display recommendations
        st.subheader("Recommended Businesses")
        
        # Display as cards in grid
        cols = st.columns(3)
        
        for i, rec in enumerate(recommendations):
            business_id = rec['business_id']
            predicted_rating = rec['predicted_rating']
            
            # Get business details
            business_details = get_business_details(business_id, business_df)
            
            # Format categories
            categories = business_details['categories']
            if isinstance(categories, str) and categories != 'Unknown':
                try:
                    categories = eval(categories)  # Convert string representation of list to actual list
                    categories = ", ".join(categories) if isinstance(categories, list) else categories
                except:
                    pass
            
            with cols[i % 3]:
                st.markdown(f"""
                <div style="border:1px solid #dddddd; padding:15px; border-radius:10px; margin-bottom:20px; background-color:#f8f9fa;">
                    <h3 style="color:#1e88e5;">{business_details['name'] if business_details['name'] != 'Unknown' else 'Business ' + business_id[:8] + '...'}</h3>
                    <p><strong>Predicted Rating:</strong> ⭐ {predicted_rating:.2f}/5.0</p>
                    <p><strong>Actual Rating:</strong> {'⭐ ' + str(business_details['stars']) + '/5.0' if business_details['stars'] != 'Unknown' else 'Unknown'}</p>
                    <p><strong>Location:</strong> {business_details['city']}, {business_details['state']}</p>
                    <p><strong>Categories:</strong> {categories}</p>
                    <p><strong>Business ID:</strong> {business_id}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Add explanation of recommendation method
        st.header("How the Recommendation Works")
        
        if method == 'hybrid':
            st.write("""
            ### Hybrid Collaborative Filtering
            
            This recommendation combines both user-based and item-based collaborative filtering approaches 
            to provide balanced recommendations that consider both similar users and similar items.
            
            - **User-based component**: Finds users with similar taste profiles and recommends items they liked
            - **Item-based component**: Finds businesses similar to those the user has highly rated
            - **Final score**: The average of both approaches
            """)
        elif method == 'user':
            st.write("""
            ### User-Based Collaborative Filtering
            
            This recommendation method finds users with similar taste profiles to the selected user,
            then recommends businesses that these similar users have rated highly.
            
            The system calculates similarity between users based on their rating patterns.
            """)
        else:  # item-based
            st.write("""
            ### Item-Based Collaborative Filtering
            
            This recommendation method finds businesses similar to those the user has already rated highly.
            
            The system calculates similarity between businesses based on how users have rated them.
            """)
        
        # Display evaluation metrics
        st.header("Model Performance Metrics")
        
        metrics_data = {
            "Metric": ["RMSE", "MAE", "Correlation"],
            "User-based": [1.63, 1.17, 0.17],
            "Item-based": [2.39, 1.75, 0.04],
            "Hybrid": [1.68, 1.30, 0.13]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create a bar chart for the metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bar_width = 0.25
        x = np.arange(len(metrics_df["Metric"]))
        
        ax.bar(x - bar_width, metrics_df["User-based"], width=bar_width, label="User-based")
        ax.bar(x, metrics_df["Item-based"], width=bar_width, label="Item-based")
        ax.bar(x + bar_width, metrics_df["Hybrid"], width=bar_width, label="Hybrid")
        
        ax.set_ylabel("Value")
        ax.set_title("Model Performance Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df["Metric"])
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("""
        - **RMSE (Root Mean Square Error)**: Lower is better. Measures prediction accuracy.
        - **MAE (Mean Absolute Error)**: Lower is better. Average absolute difference between predicted and actual ratings.
        - **Correlation**: Higher is better. How well predicted ratings correlate with actual ratings.
        """)

if __name__ == "__main__":
    main()