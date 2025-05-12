import os
import sys
import streamlit as st
import torch
import pandas as pd

# Add the project directory to the Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)

from inference import YelpRecommendationInference

# Cached model path
MODEL_PATH = os.path.join(project_dir, 'checkpoints', 'best_model.pt')
CACHE_DIR = os.path.join(project_dir, 'cache')

def get_user_list():
    """Fetch list of users from Neo4j."""
    inference = YelpRecommendationInference(MODEL_PATH, cache_dir=CACHE_DIR)
    
    try:
        with inference.data_loader.driver.session() as session:
            query = """
            MATCH (u:User)
            RETURN u.user_id AS user_id, 
                   u.name AS name,
                   u.review_count AS review_count
            ORDER BY u.review_count DESC
            LIMIT 100  
            """
            result = session.run(query)
            users = [
                {
                    'user_id': record['user_id'], 
                    'name': record['name'] or record['user_id'],
                    'review_count': record['review_count']
                } 
                for record in result
            ]
    finally:
        inference.close()
    
    return users

def get_business_list():
    """Fetch list of businesses from Neo4j."""
    inference = YelpRecommendationInference(MODEL_PATH, cache_dir=CACHE_DIR)
    
    try:
        with inference.data_loader.driver.session() as session:
            query = """
            MATCH (b:Business)
            RETURN b.business_id AS business_id, 
                   b.name AS name,
                   b.review_count AS review_count
            ORDER BY b.review_count DESC
            LIMIT 100
            """
            result = session.run(query)
            businesses = [
                {
                    'business_id': record['business_id'], 
                    'name': record['name'] or record['business_id'],
                    'review_count': record['review_count']
                } 
                for record in result
            ]
    finally:
        inference.close()
    
    return businesses

def display_recommendations(recommendations, recommendation_type):
    """
    Display recommendations in a styled Streamlit layout.
    
    Args:
        recommendations: DataFrame of recommendations
        recommendation_type: 'recommendations' or 'similar_businesses'
    """
    # Create a container for recommendations
    st.markdown("### 🌟 Recommendations")
    
    # Check if recommendations are empty
    if len(recommendations) == 0:
        st.warning("No recommendations found.")
        return
    
    # Determine the column names based on recommendation type
    score_column = 'predicted_score' if recommendation_type == 'recommendations' else 'similarity'
    
    # Create individual cards for each recommendation
    for _, row in recommendations.iterrows():
        with st.container():
            # Business name as header
            st.markdown(f"### {row['name']}")
            
            # Create columns for details
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Basic business info
                st.metric("Stars", f"{row.get('stars', 'N/A'):.1f}")
            
            with col2:
                # Review count
                st.metric("Review Count", row.get('review_count', 'N/A'))
            
            with col3:
                # Recommendation score
                st.metric(
                    "Score" if recommendation_type == 'recommendations' else "Similarity", 
                    f"{row[score_column]:.2f}"
                )
            
            # Business categories
            st.markdown(f"**Categories:** {row.get('categories', 'N/A')}")
            
            # Separator
            st.markdown("---")

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Yelp Recommendation System", 
        page_icon="🍽️", 
        layout="wide"
    )
    
    # Title and description
    st.title("🍽️ Yelp Business Recommendation System")
    st.markdown("""
    Discover personalized business recommendations and find similar businesses 
    using advanced Graph Neural Network (GNN) techniques.
    """)
    
    # Sidebar for navigation
    st.sidebar.header("🔍 Recommendation Options")
    
    # Choose recommendation type
    rec_type = st.sidebar.radio(
        "Select Recommendation Type", 
        ["User Recommendations", "Similar Businesses"]
    )
    
    # Load users and businesses
    try:
        users = get_user_list()
        businesses = get_business_list()
    except Exception as e:
        st.error(f"Error connecting to Neo4j: {e}")
        return
    
    # Initialize inference engine
    try:
        inference = YelpRecommendationInference(MODEL_PATH, cache_dir=CACHE_DIR)
    except Exception as e:
        st.error(f"Error initializing recommendation engine: {e}")
        return
    
    # User Recommendations
    if rec_type == "User Recommendations":
        # Create user selection with additional info
        user_options = {
            user['user_id']: f"{user['name']} (Reviews: {user['review_count']})" 
            for user in users
        }
        selected_user_id = st.sidebar.selectbox(
            "Select a User", 
            list(user_options.keys()), 
            format_func=lambda x: user_options.get(x, x)
        )
        
        # Number of recommendations
        top_k = st.sidebar.slider("Number of Recommendations", 5, 20, 10)
        
        # Generate recommendations button
        if st.sidebar.button("Get Recommendations"):
            try:
                # Get recommendations
                recommendations = inference.get_recommendations(
                    user_id=selected_user_id, 
                    top_k=top_k
                )
                
                # Display user details
                user_details = next(user for user in users if user['user_id'] == selected_user_id)
                st.markdown(f"### 👤 User: {user_details['name']}")
                st.markdown(f"**Total Reviews:** {user_details['review_count']}")
                
                # Display recommendations
                display_recommendations(recommendations, 'recommendations')
            
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
    
    # Similar Businesses
    else:
        # Create business selection with additional info
        business_options = {
            business['business_id']: f"{business['name']} (Reviews: {business['review_count']})" 
            for business in businesses
        }
        selected_business_id = st.sidebar.selectbox(
            "Select a Business", 
            list(business_options.keys()), 
            format_func=lambda x: business_options.get(x, x)
        )
        
        # Number of similar businesses
        top_k = st.sidebar.slider("Number of Similar Businesses", 5, 20, 10)
        
        # Generate similar businesses button
        if st.sidebar.button("Find Similar Businesses"):
            try:
                # Get similar businesses
                similar_businesses = inference.get_similar_businesses(
                    business_id=selected_business_id, 
                    top_k=top_k
                )
                
                # Display base business details
                base_business = next(business for business in businesses if business['business_id'] == selected_business_id)
                st.markdown(f"### 🏢 Base Business: {base_business['name']}")
                st.markdown(f"**Total Reviews:** {base_business['review_count']}")
                st.markdown(f"**Categories:** {base_business.get('categories', 'N/A')}")
                
                # Display similar businesses
                display_recommendations(similar_businesses, 'similar_businesses')
            
            except Exception as e:
                st.error(f"Error finding similar businesses: {e}")
    
    # Close resources
    inference.close()

if __name__ == "__main__":
    main()