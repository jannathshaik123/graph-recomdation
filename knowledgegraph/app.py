import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import YelpRecommendationSystem, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
import os
import time
import pickle
import json
from datetime import datetime

# Add this line to the initialization section at the beginning of the file
if 'current_model_timestamp' not in st.session_state:
    st.session_state.current_model_timestamp = None
    
# Initialize session state variables if they don't exist
if 'recommender' not in st.session_state:
    st.session_state.recommender = YelpRecommendationSystem(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

if 'user_factors' not in st.session_state:
    st.session_state.user_factors = None

if 'business_factors' not in st.session_state:
    st.session_state.business_factors = None

if 'global_avg' not in st.session_state:
    st.session_state.global_avg = None

if 'available_users' not in st.session_state:
    st.session_state.available_users = []

if 'available_businesses' not in st.session_state:
    st.session_state.available_businesses = []

if 'metrics' not in st.session_state:
    st.session_state.metrics = {}

if 'test_set' not in st.session_state:
    st.session_state.test_set = []


def load_model(timestamp=None):
    """Load a trained model or train a new one if requested"""
    with st.spinner("Loading model..."):
        user_factors, business_factors, global_avg = st.session_state.recommender.load_matrix_factorization_model(timestamp=timestamp)
        
        if user_factors is not None:
            st.session_state.user_factors = user_factors
            st.session_state.business_factors = business_factors
            st.session_state.global_avg = global_avg
            st.success(f"Model loaded successfully with {len(user_factors)} users and {len(business_factors)} businesses")
            return True
        else:
            st.error("Failed to load model")
            return False


def train_new_model():
    """Train a new recommendation model"""
    with st.spinner("Training new model..."):
        # Split data into training and test sets
        train_set, test_set = st.session_state.recommender.split_train_test(
            test_size=0.2, 
            min_reviews=3, 
            sample_size=st.session_state.sample_size
        )
        
        if not train_set or not test_set:
            st.error("Insufficient data for training")
            return False
        
        st.session_state.test_set = test_set
        
        # Train matrix factorization model
        user_factors, business_factors, global_avg = st.session_state.recommender.train_matrix_factorization(
            num_factors=st.session_state.num_factors,
            learning_rate=st.session_state.learning_rate,
            regularization=st.session_state.reg_factor,
            num_iterations=st.session_state.num_iterations,
            sample_size=len(train_set)
        )
        
        # Save the trained model
        model_info = st.session_state.recommender.save_matrix_factorization_model(user_factors, business_factors, global_avg)
        
        # Update session state
        st.session_state.user_factors = user_factors
        st.session_state.business_factors = business_factors
        st.session_state.global_avg = global_avg
        
        # Evaluate model
        metrics = st.session_state.recommender.evaluate_recommendations(
            test_set, user_factors, business_factors, global_avg
        )
        st.session_state.metrics = metrics
        
        st.success(f"Model trained successfully with {len(user_factors)} users and {len(business_factors)} businesses")
        return True


def load_available_users(limit=100):
    """Load available users for recommendations"""
    with st.spinner("Loading users..."):
        # Execute a Cypher query to get users with reviews
        with st.session_state.recommender.driver.session() as session:
            query = """
            MATCH (u:User)-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
            WITH u, count(r) as num_reviews
            WHERE num_reviews >= 3
            RETURN u.user_id as user_id, u.name as name, num_reviews
            ORDER BY num_reviews DESC
            LIMIT $limit
            """
            result = session.run(query, limit=limit)
            users = [{"user_id": record["user_id"], 
                      "name": record["name"],
                      "num_reviews": record["num_reviews"]} for record in result]
        
        st.session_state.available_users = users
        return users


def load_available_businesses(limit=100):
    """Load available businesses for evaluation"""
    with st.spinner("Loading businesses..."):
        # Execute a Cypher query to get businesses with reviews
        with st.session_state.recommender.driver.session() as session:
            query = """
            MATCH (b:Business)<-[:ABOUT]-(r:Review)
            WITH b, count(r) as num_reviews
            WHERE num_reviews >= 5
            RETURN b.business_id as business_id, b.name as name, b.stars as avg_stars, num_reviews
            ORDER BY num_reviews DESC
            LIMIT $limit
            """
            result = session.run(query, limit=limit)
            businesses = [{"business_id": record["business_id"], 
                          "name": record["name"],
                          "avg_stars": record["avg_stars"],
                          "num_reviews": record["num_reviews"]} for record in result]
        
        st.session_state.available_businesses = businesses
        return businesses


def display_user_profile(user_id):
    """Display user profile information"""
    # Get user details
    with st.session_state.recommender.driver.session() as session:
        query = """
        MATCH (u:User {user_id: $user_id})
        RETURN u.name as name, u.review_count as review_count, 
               u.average_stars as avg_stars, u.yelping_since as yelping_since,
               u.fans as fans, u.useful as useful
        """
        result = session.run(query, user_id=user_id)
        user_details = result.single()
        
        if user_details:
            st.subheader(f"User Profile: {user_details['name']} (ID: {user_id})")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Rating", f"{user_details['avg_stars']:.1f}⭐")
            with col2:
                st.metric("Reviews", user_details['review_count'])
            with col3:
                st.metric("Fans", user_details['fans'])
            
            # Get recent reviews
            query = """
            MATCH (u:User {user_id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
            RETURN b.name as business_name, r.stars as rating, r.date as date
            ORDER BY r.date DESC
            LIMIT 5
            """
            result = session.run(query, user_id=user_id)
            reviews = [dict(record) for record in result]
            
            if reviews:
                st.subheader("Recent Reviews")
                for review in reviews:
                    st.write(f"**{review['business_name']}** - {review['rating']}⭐ ({review['date'][:10]})")
        else:
            st.warning(f"User with ID {user_id} not found")


def display_business_profile(business_id):
    """Display business profile information"""
    business_details = st.session_state.recommender.get_business_details(business_id)
    
    if business_details:
        st.subheader(f"Business: {business_details['name']} (ID: {business_id})")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Rating", f"{business_details['avg_stars']:.1f}⭐")
        with col2:
            st.metric("Reviews", business_details['review_count'])
        with col3:
            st.metric("City", business_details['city'])
        
        # Display categories
        if business_details['categories']:
            st.write("**Categories:** " + ", ".join(business_details['categories']))
    else:
        st.warning(f"Business with ID {business_id} not found")


def get_recommendations(user_id, rec_type, top_n):
    """Get recommendations based on selected method"""
    with st.spinner(f"Getting {rec_type} recommendations..."):
        if rec_type == "Collaborative Filtering":
            return st.session_state.recommender.collaborative_filtering_recommendations(user_id, top_n=top_n)
        elif rec_type == "Content-Based":
            return st.session_state.recommender.content_based_recommendations(user_id, top_n=top_n)
        elif rec_type == "Hybrid":
            return st.session_state.recommender.hybrid_recommendations(user_id, top_n=top_n)
        else:
            st.error("Invalid recommendation type")
            return []


def model_evaluation_page():
    """Page for evaluating recommendation models"""
    st.title("Model Evaluation")
    
    # Load available models
    available_models = st.session_state.recommender.get_available_models()
    
    if not available_models:
        st.warning("No trained models available. Please train a model first.")
        return
    
    # Create tabs for different evaluation views
    tabs = st.tabs(["Model Performance", "User-Item Predictions", "Train New Model"])
    
    with tabs[0]:  # Model Performance tab
        st.subheader("Model Performance Metrics")
        
        # Model selection dropdown
        model_options = [f"Model from {model['timestamp']} - {model['num_users']} users" for model in available_models]
        selected_model_idx = st.selectbox("Select Model", range(len(model_options)), format_func=lambda x: model_options[x])
        
        selected_model = available_models[selected_model_idx]
        
        # Load selected model if not already loaded
        if st.session_state.user_factors is None or selected_model['timestamp'] != st.session_state.current_model_timestamp:
            if load_model(selected_model['timestamp']):
                st.session_state.current_model_timestamp = selected_model['timestamp']
                
                # Get test set for evaluation if not available
                if not st.session_state.test_set:
                    _, test_set = st.session_state.recommender.split_train_test(test_size=0.2, min_reviews=3, sample_size=10000)
                    st.session_state.test_set = test_set
                
                # Evaluate model
                metrics = st.session_state.recommender.evaluate_recommendations(
                    st.session_state.test_set, st.session_state.user_factors, 
                    st.session_state.business_factors, st.session_state.global_avg
                )
                st.session_state.metrics = metrics
        
        # Display metrics if available
        if st.session_state.metrics:
            metrics = st.session_state.metrics
            
            # Create a comparison table
            df_metrics = pd.DataFrame({
                'Model': ['Baseline', 'Matrix Factorization'],
                'MAE': [metrics['baseline']['mae'], metrics['matrix_factorization']['mae']],
                'RMSE': [metrics['baseline']['rmse'], metrics['matrix_factorization']['rmse']]
            })
            
            st.dataframe(df_metrics)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(df_metrics))
            width = 0.35
            
            ax.bar(x - width/2, df_metrics['MAE'], width, label='MAE')
            ax.bar(x + width/2, df_metrics['RMSE'], width, label='RMSE')
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Error')
            ax.set_title('Model Evaluation Metrics')
            ax.set_xticks(x)
            ax.set_xticklabels(df_metrics['Model'])
            ax.legend()
            
            st.pyplot(fig)
            
            # Model details
            st.subheader("Model Details")
            st.write(f"**Global Average Rating:** {st.session_state.global_avg:.2f}")
            st.write(f"**Number of Users:** {len(st.session_state.user_factors)}")
            st.write(f"**Number of Businesses:** {len(st.session_state.business_factors)}")
            st.write(f"**Number of Factors:** {selected_model['factors_shape']}")
        else:
            st.info("No evaluation metrics available. Please train or load a model.")
    
    with tabs[1]:  # User-Item Predictions tab
        st.subheader("User-Item Rating Predictions")
        
        # Load users and businesses if not already loaded
        if not st.session_state.available_users:
            st.session_state.available_users = load_available_users()
        
        if not st.session_state.available_businesses:
            st.session_state.available_businesses = load_available_businesses()
        
        # Create selection boxes with combined name and ID
        user_options = [f"{user['name']} (ID: {user['user_id']}) - {user['num_reviews']} reviews" 
                      for user in st.session_state.available_users]
        
        business_options = [f"{business['name']} (ID: {business['business_id']}) - {business['avg_stars']}⭐" 
                          for business in st.session_state.available_businesses]
        
        # User and business selection
        col1, col2 = st.columns(2)
        
        with col1:
            selected_user_idx = st.selectbox("Select User", range(len(user_options)), format_func=lambda x: user_options[x])
            selected_user = st.session_state.available_users[selected_user_idx]
            user_id = selected_user['user_id']
            
        with col2:
            selected_business_idx = st.selectbox("Select Business", range(len(business_options)), format_func=lambda x: business_options[x])
            selected_business = st.session_state.available_businesses[selected_business_idx]
            business_id = selected_business['business_id']
        
        # Display user and business profiles
        display_user_profile(user_id)
        st.markdown("---")
        display_business_profile(business_id)
        
        # Make predictions
        if st.session_state.user_factors is not None and st.session_state.business_factors is not None:
            st.markdown("---")
            st.subheader("Rating Predictions")
            
            # Get actual rating if available
            actual_rating = st.session_state.recommender.get_rating(user_id, business_id)
            
            # Calculate predictions
            baseline_pred = st.session_state.recommender.baseline_predict(user_id, business_id)
            
            mf_pred = None
            if (user_id in st.session_state.user_factors and 
                business_id in st.session_state.business_factors):
                user_vector = st.session_state.user_factors[user_id]
                business_vector = st.session_state.business_factors[business_id]
                
                # Extract bias terms
                user_bias = user_vector[-1]
                business_bias = business_vector[-1]
                
                # Dot product of feature vectors
                dot_product = np.dot(user_vector[:-1], business_vector[:-1])
                
                # Final prediction
                mf_pred = st.session_state.global_avg + user_bias + business_bias + dot_product
                mf_pred = max(1.0, min(5.0, mf_pred))  # Clip to valid range
            
            # Display predictions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Actual Rating", f"{actual_rating:.1f}⭐" if actual_rating else "Not Rated")
            
            with col2:
                st.metric("Baseline Prediction", f"{baseline_pred:.1f}⭐")
                
            with col3:
                if mf_pred is not None:
                    st.metric("Matrix Factorization", f"{mf_pred:.1f}⭐")
                else:
                    st.info("No MF prediction available")
            
            # Explanation
            st.markdown("---")
            st.subheader("Prediction Explanation")
            
            st.write("**Baseline Prediction Components:**")
            global_avg = st.session_state.global_avg
            user_avg = st.session_state.recommender.get_user_average_rating(user_id)
            business_avg = st.session_state.recommender.get_business_average_rating(business_id)
            
            st.write(f"- Global Average: {global_avg:.2f}")
            st.write(f"- User Bias: {user_avg - global_avg:.2f}")
            st.write(f"- Business Bias: {business_avg - global_avg:.2f}")
            st.write(f"- Final Baseline Prediction: {baseline_pred:.2f}")
            
            if mf_pred is not None:
                st.write("**Matrix Factorization Components:**")
                st.write(f"- Global Average: {global_avg:.2f}")
                st.write(f"- User Bias: {user_bias:.2f}")
                st.write(f"- Business Bias: {business_bias:.2f}")
                st.write(f"- User-Business Interaction: {dot_product:.2f}")
                st.write(f"- Final MF Prediction: {mf_pred:.2f}")
    
    with tabs[2]:  # Train New Model tab
        st.subheader("Train New Model")
        
        # Model parameters
        st.session_state.num_factors = st.slider("Number of Factors", 5, 50, 15)
        st.session_state.learning_rate = st.number_input("Learning Rate", 0.001, 0.1, 0.005, format="%.3f")
        st.session_state.reg_factor = st.number_input("Regularization Factor", 0.001, 0.1, 0.02, format="%.3f")
        st.session_state.num_iterations = st.slider("Number of Iterations", 5, 100, 20)
        st.session_state.sample_size = st.slider("Sample Size", 5000, 100000, 50000, step=5000)
        
        # Train button
        if st.button("Train New Model"):
            if train_new_model():
                st.success("Model trained successfully! You can evaluate it in the Model Performance tab.")


def recommendation_page():
    """Page for generating recommendations for users"""
    st.title("Yelp Recommendation System")
    
    # Check if model is loaded
    if st.session_state.user_factors is None:
        # Load latest model if available
        available_models = st.session_state.recommender.get_available_models()
        if available_models:
            st.info("Loading the latest model...")
            load_model()
        else:
            st.warning("No trained models available. Please go to Model Evaluation and train a model first.")
            return
    
    # Load users if not already loaded
    if not st.session_state.available_users:
        st.session_state.available_users = load_available_users()
    
    # User selection with combined name and ID
    user_options = [f"{user['name']} (ID: {user['user_id']}) - {user['num_reviews']} reviews" 
                  for user in st.session_state.available_users]
    
    selected_user_idx = st.selectbox("Select User", range(len(user_options)), format_func=lambda x: user_options[x])
    selected_user = st.session_state.available_users[selected_user_idx]
    user_id = selected_user['user_id']
    
    # Display user profile
    display_user_profile(user_id)
    
    # Recommendation options
    st.sidebar.header("Recommendation Options")
    rec_type = st.sidebar.radio("Recommendation Type", 
                               ["Collaborative Filtering", "Content-Based", "Hybrid"])
    top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10)
    
    # Get recommendations button
    if st.button("Get Recommendations"):
        recommendations = get_recommendations(user_id, rec_type, top_n)
        
        if recommendations:
            st.subheader(f"{rec_type} Recommendations for {selected_user['name']}")
            
            # Display recommendations in cards
            for i, rec in enumerate(recommendations):
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        if 'predicted_rating' in rec:
                            st.subheader(f"{i+1}. {rec['name']} - {rec['predicted_rating']:.1f}⭐")
                        else:
                            st.subheader(f"{i+1}. {rec['name']} - {rec['score']:.2f} (score)")
                            
                        # Show categories
                        if 'categories' in rec and rec['categories']:
                            st.write("**Categories:** " + ", ".join(rec['categories'][:5]))
                        
                        # Show city if available
                        if 'city' in rec and rec['city']:
                            st.write(f"**Location:** {rec['city']}")
                    
                    with col2:
                        if 'avg_stars' in rec:
                            st.metric("Avg Rating", f"{rec['avg_stars']:.1f}⭐")
                        
                        # Add a button to view more details
                        if st.button(f"Details #{i+1}", key=f"btn_{i}"):
                            display_business_profile(rec['business_id'])
                    
                    st.markdown("---")
        else:
            st.warning("No recommendations found for this user.")


def main():
    st.set_page_config(
        page_title="Yelp Recommendation System",
        page_icon="🍽️",
        layout="wide"
    )
    
    # Add module reloading button to sidebar
    with st.sidebar:
        if st.button("Reload Modules"):
            import importlib
            import utils  # Assuming this is your module with YelpRecommendationSystem
            importlib.reload(utils)
            st.success("Backend modules reloaded!")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Recommendations", "Model Evaluation"])
    
    # Display selected page
    if page == "Recommendations":
        recommendation_page()
    else:
        model_evaluation_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("Yelp Recommendation System powered by Neo4j")


if __name__ == "__main__":
    main()