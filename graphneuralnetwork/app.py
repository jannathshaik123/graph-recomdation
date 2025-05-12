import os
import streamlit as st
import torch
import pandas as pd
import sys

# Add the project directory to the Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)

from inference import YelpRecommendationInference

# Neo4j connection configuration
NEO4J_URI = "bolt://localhost:7687"  # Update if your Neo4j instance is hosted elsewhere
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # Change this to your actual password

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
                   u.name AS name 
            LIMIT 100  # Limit to prevent overwhelming the dropdown
            """
            result = session.run(query)
            users = [{'user_id': record['user_id'], 'name': record['name'] or record['user_id']} for record in result]
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
                   b.name AS name 
            LIMIT 100  # Limit to prevent overwhelming the dropdown
            """
            result = session.run(query)
            businesses = [{'business_id': record['business_id'], 'name': record['name'] or record['business_id']} for record in result]
    finally:
        inference.close()
    
    return businesses

def main():
    # Set page configuration
    st.set_page_config(page_title="Yelp Recommendation System", page_icon=":star:", layout="wide")
    
    # Title
    st.title("🌟 Yelp Recommendation System")
    
    # Sidebar for navigation
    st.sidebar.header("Recommendation Options")
    
    # Choose recommendation type
    rec_type = st.sidebar.radio("Select Recommendation Type", 
                                ["User Recommendations", "Similar Businesses"])
    
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
        st.subheader("Get Personalized Business Recommendations")
        
        # User selection
        user_options = {user['user_id']: user['name'] for user in users}
        selected_user_id = st.selectbox(
            "Select a User", 
            list(user_options.keys()), 
            format_func=lambda x: user_options.get(x, x)
        )
        
        # Number of recommendations
        top_k = st.slider("Number of Recommendations", 5, 20, 10)
        
        # Generate recommendations button
        if st.button("Get Recommendations"):
            try:
                # Get recommendations
                recommendations = inference.get_recommendations(
                    user_id=selected_user_id, 
                    top_k=top_k
                )
                
                # Display recommendations
                st.subheader(f"Top {top_k} Recommendations for {user_options.get(selected_user_id, selected_user_id)}")
                
                # Create columns for recommendations
                for _, row in recommendations.iterrows():
                    with st.container():
                        st.markdown(f"**{row['name']}**")
                        st.markdown(f"Stars: {row.get('stars', 'N/A')}")
                        st.markdown(f"Review Count: {row.get('review_count', 'N/A')}")
                        st.markdown(f"Categories: {row.get('categories', 'N/A')}")
                        st.markdown(f"Predicted Score: {row['predicted_score']:.2f}")
                        st.markdown("---")
            
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
    
    # Similar Businesses
    else:
        st.subheader("Find Similar Businesses")
        
        # Business selection
        business_options = {business['business_id']: business['name'] for business in businesses}
        selected_business_id = st.selectbox(
            "Select a Business", 
            list(business_options.keys()), 
            format_func=lambda x: business_options.get(x, x)
        )
        
        # Number of similar businesses
        top_k = st.slider("Number of Similar Businesses", 5, 20, 10)
        
        # Generate similar businesses button
        if st.button("Find Similar Businesses"):
            try:
                # Get similar businesses
                similar_businesses = inference.get_similar_businesses(
                    business_id=selected_business_id, 
                    top_k=top_k
                )
                
                # Display similar businesses
                st.subheader(f"Top {top_k} Businesses Similar to {business_options.get(selected_business_id, selected_business_id)}")
                
                # Create columns for similar businesses
                for _, row in similar_businesses.iterrows():
                    with st.container():
                        st.markdown(f"**{row['name']}**")
                        st.markdown(f"Stars: {row.get('stars', 'N/A')}")
                        st.markdown(f"Review Count: {row.get('review_count', 'N/A')}")
                        st.markdown(f"Categories: {row.get('categories', 'N/A')}")
                        st.markdown(f"Similarity Score: {row['similarity']:.2f}")
                        st.markdown("---")
            
            except Exception as e:
                st.error(f"Error finding similar businesses: {e}")
    
    # Close resources
    inference.close()

if __name__ == "__main__":
    main()