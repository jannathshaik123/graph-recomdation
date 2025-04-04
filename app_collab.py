import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from scipy.sparse import load_npz
import sys
import matplotlib.pyplot as plt

# Import the collaborative filtering class from the provided code
# You'll need to make sure this file is in the same directory
from collaborative_filtering import MemoryEfficientCF, load_ratings_matrix

st.set_page_config(page_title="Restaurant Recommender System", layout="wide")

# App title and description
st.title("Restaurant Recommendation System")
st.markdown("""
This application demonstrates a memory-efficient collaborative filtering recommendation system.
Select a user to see personalized restaurant recommendations based on their previous ratings and similar users' preferences.
""")

# Function to load the model and necessary data
@st.cache_resource
def load_model_and_data(model_dir=os.path.join(os.path.dirname(os.getcwd()),"model")):
    """Load the trained model and additional business data"""
    try:
        # Load the collaborative filtering model
        model = MemoryEfficientCF.load_model(model_dir)
        ratings_matrix = load_ratings_matrix(os.path.join(model_dir, "ratings_matrix.npz"))
        
        # Load business metadata (we'll simulate this data)
        # In a real application, you would load this from your dataset
        try:
            business_metadata = pd.read_csv('business_metadata.csv')
        except FileNotFoundError:
            # Create dummy metadata if file doesn't exist
            st.warning("Business metadata file not found. Creating dummy data for demonstration.")
            business_ids = list(model.idx_to_business.values())
            business_metadata = pd.DataFrame({
                'business_id': business_ids,
                'name': [f"Restaurant {i+1}" for i in range(len(business_ids))],
                'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Miami', 'Seattle'], len(business_ids)),
                'state': np.random.choice(['NY', 'CA', 'IL', 'FL', 'WA'], len(business_ids)),
                'stars': np.random.uniform(1, 5, len(business_ids)).round(1),
                'review_count': np.random.randint(10, 1000, len(business_ids))
            })
            business_metadata.to_csv('business_metadata.csv', index=False)
            
        # Load user metadata (we'll simulate this data)
        try:
            user_metadata = pd.read_csv('user_metadata.csv')
        except FileNotFoundError:
            # Create dummy metadata if file doesn't exist
            user_ids = list(model.idx_to_user.values())
            user_metadata = pd.DataFrame({
                'user_id': user_ids,
                'name': [f"User {i+1}" for i in range(len(user_ids))],
                'review_count': np.random.randint(1, 100, len(user_ids)),
                'average_stars': np.random.uniform(1, 5, len(user_ids)).round(1)
            })
            user_metadata.to_csv('user_metadata.csv', index=False)
            
        return model, ratings_matrix, business_metadata, user_metadata
    except Exception as e:
        st.error(f"Error loading model and data: {e}")
        return None, None, None, None

# Load the model and data
model, ratings_matrix, business_metadata, user_metadata = load_model_and_data()

# Check if data loaded successfully
if model is None or ratings_matrix is None:
    st.error("Failed to load recommendation model. Please check that the model directory exists.")
    st.stop()

# Create a sidebar for user selection
st.sidebar.header("User Selection")

# Get list of users
user_list = list(model.user_to_idx.keys())
user_ids_with_names = []

# Match user IDs with names when possible
if user_metadata is not None:
    for user_id in user_list[:100]:  # Limit to first 100 for performance
        user_name = "Unknown"
        user_info = user_metadata[user_metadata['user_id'] == user_id]
        if not user_info.empty:
            user_name = user_info.iloc[0]['name']
        user_ids_with_names.append(f"{user_name} ({user_id[:8]}...)")

# If we couldn't create the combined list, just use IDs
if not user_ids_with_names:
    # Truncate IDs for display
    user_ids_with_names = [f"User ({user_id[:8]}...)" for user_id in user_list[:100]]

# Create a selection box with search
selected_user_display = st.sidebar.selectbox(
    "Select a user:",
    options=user_ids_with_names
)

# Extract the actual user ID from the display string
selected_user_id = user_list[user_ids_with_names.index(selected_user_display)]

# Method selection
st.sidebar.header("Recommendation Method")
method = st.sidebar.radio(
    "Select recommendation method:",
    options=["hybrid", "user", "item"],
    help="User-based uses similar users' ratings. Item-based uses similar items. Hybrid combines both approaches."
)

# Number of recommendations
num_recommendations = st.sidebar.slider("Number of recommendations:", 1, 20, 5)

# Function to generate recommendations
def get_recommendations(user_id, n=5, method='hybrid'):
    """Generate recommendations for the selected user"""
    if user_id not in model.user_to_idx:
        return pd.DataFrame()
    
    user_idx = model.user_to_idx[user_id]
    recommended_items = model.recommend_top_n(ratings_matrix, user_idx, n=n, method=method)
    
    # Create recommendations dataframe
    recommendations = []
    for item_idx, predicted_rating in recommended_items:
        business_id = model.idx_to_business[item_idx]
        
        # Get business info
        business_info = {"business_id": business_id, "predicted_rating": predicted_rating}
        
        # Add metadata if available
        if business_metadata is not None:
            business_data = business_metadata[business_metadata['business_id'] == business_id]
            if not business_data.empty:
                business = business_data.iloc[0]
                business_info["name"] = business['name']
                business_info["city"] = business['city']
                business_info["state"] = business['state']
                business_info["avg_stars"] = business['stars']
                business_info["review_count"] = business['review_count']
            else:
                business_info["name"] = "Unknown"
                business_info["city"] = "Unknown"
                business_info["state"] = "Unknown"
                business_info["avg_stars"] = 0
                business_info["review_count"] = 0
        
        recommendations.append(business_info)
    
    return pd.DataFrame(recommendations)

# Get user info
user_info = None
if user_metadata is not None:
    user_data = user_metadata[user_metadata['user_id'] == selected_user_id]
    if not user_data.empty:
        user_info = user_data.iloc[0]

# Display user information
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("User Information")
    if user_info is not None:
        st.write(f"**Name:** {user_info['name']}")
        st.write(f"**User ID:** {selected_user_id}")
        st.write(f"**Review Count:** {user_info['review_count']}")
        st.write(f"**Average Rating:** {user_info['average_stars']:.1f} ⭐")
    else:
        st.write(f"**User ID:** {selected_user_id}")
        st.write("No additional user information available.")

# Get recommendations when user clicks the button
if st.button("Generate Recommendations"):
    with st.spinner("Generating recommendations..."):
        recommendations_df = get_recommendations(selected_user_id, n=num_recommendations, method=method)
    
    # Display recommendations
    if not recommendations_df.empty:
        st.subheader(f"Top {num_recommendations} Recommended Restaurants")
        
        # Display recommendations in cards
        for i, (_, rec) in enumerate(recommendations_df.iterrows()):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Display prediction score as a gauge chart
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.pie([rec['predicted_rating'], 5-rec['predicted_rating']], 
                       colors=['#1f77b4', '#f0f0f0'], 
                       startangle=90, 
                       counterclock=False,
                       wedgeprops={'width': 0.3, 'edgecolor': 'w'})
                ax.text(0, 0, f"{rec['predicted_rating']:.1f}", ha='center', va='center', fontsize=20)
                ax.set_title("Predicted Rating", pad=10)
                ax.axis('equal')
                st.pyplot(fig)

            with col2:
                if 'name' in rec:
                    st.markdown(f"### {i+1}. {rec['name']}")
                else:
                    st.markdown(f"### {i+1}. Restaurant {rec['business_id'][:8]}...")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if 'city' in rec and 'state' in rec:
                        st.write(f"📍 {rec['city']}, {rec['state']}")
                    else:
                        st.write("📍 Location unknown")
                
                with col_b:
                    if 'avg_stars' in rec:
                        st.write(f"⭐ {rec['avg_stars']:.1f} (avg rating)")
                    else:
                        st.write("⭐ No ratings")
                        
                with col_c:
                    if 'review_count' in rec:
                        st.write(f"📝 {rec['review_count']} reviews")
                    else:
                        st.write("📝 No reviews")
                
                st.write(f"**Business ID:** {rec['business_id']}")
                st.write(f"**Predicted Rating:** {rec['predicted_rating']:.2f}")
            
            st.divider()
    else:
        st.warning("No recommendations could be generated for this user.")

# Display explanations of the recommendation algorithms
with st.expander("How do the recommendation methods work?"):
    st.markdown("""
    ### Recommendation Methods
    
    **User-based Collaborative Filtering**
    - Finds users similar to the selected user
    - Recommends businesses that similar users liked
    - Works best when many users have rated the same items
    
    **Item-based Collaborative Filtering**
    - Finds businesses similar to those the user has rated highly
    - Recommends businesses with similar characteristics
    - Works well when user preferences are consistent
    
    **Hybrid Method**
    - Combines both user-based and item-based recommendations
    - Often provides more balanced and accurate recommendations
    - Helps mitigate the limitations of each individual approach
    """)

with st.expander("About the Evaluation Metrics"):
    st.markdown("""
    ### Model Evaluation Metrics
    
    The model was evaluated using the following metrics:
    
    - **RMSE (Root Mean Square Error)**: Measures the square root of the average squared differences between predicted and actual ratings
    - **MAE (Mean Absolute Error)**: Measures the average absolute differences between predicted and actual ratings
    - **Correlation**: Measures how well the predicted ratings correlate with actual ratings
    
    Based on the evaluation results, the **user-based** approach performed best for this dataset, followed by the **hybrid** approach.
    """)

# Footer information
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About This App
This recommendation system uses collaborative filtering to suggest restaurants based on user ratings. 
The app demonstrates both user-based and item-based approaches, as well as a hybrid method.
""")