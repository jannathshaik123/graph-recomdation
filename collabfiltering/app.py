import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from scipy.sparse import load_npz
import sys
import json
import time
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import random
import altair as alt

# Import the ImprovedCollaborativeFiltering class from utils.py
sys.path.append(".")
from utils import ImprovedCollaborativeFiltering

def load_model_and_data(model_dir="model_new"):
    """Load the trained model and data"""
    try:
        model = ImprovedCollaborativeFiltering.load_model(model_dir)
        ratings_matrix = load_npz(os.path.join(model_dir, "ratings_matrix.npz"))
        return model, ratings_matrix
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def load_business_data(file_path=r"extract\business_data.csv"):
    """Load business metadata if available"""
    if os.path.exists(file_path):
        try:
            business_df = pd.read_csv(file_path)
            # Create a dictionary for quick lookups
            business_info = {row['business_id']: row for _, row in business_df.iterrows()}
            return business_info
        except Exception as e:
            st.warning(f"Could not load business data: {e}")
    return {}

def load_user_data(file_path=r"extract\user_data.csv"):
    """Load user metadata if available"""
    user_info = {}
    if os.path.exists(file_path):
        try:
            user_df = pd.read_csv(file_path)
            # Create a dictionary for quick lookups
            user_info = {row['user_id']: row for _, row in user_df.iterrows()}
            return user_info
        except Exception as e:
            st.warning(f"Could not load user data: {e}")
    else:
        st.warning(f"User data file not found: {file_path}")
    return user_info  # Return empty dictionary if no data loaded


def get_user_ids_with_names(model, user_info, n=100):
    """Get a sample of user IDs with names for the dropdown"""
    if model and model.user_to_idx:
        # Return either all user IDs or a sample of n
        user_ids = list(model.user_to_idx.keys())
        if len(user_ids) > n:
            # Use a fixed seed for consistent sampling
            np.random.seed(42)
            sample_ids = np.random.choice(user_ids, n, replace=False).tolist()
        else:
            sample_ids = user_ids
        
        # Create display names with username if available
        user_options = []
        for user_id in sample_ids:
            display_name = user_id
            if user_id in user_info and 'name' in user_info[user_id]:
                display_name = f"{user_info[user_id]['name']} ({user_id})"
            user_options.append({"label": display_name, "value": user_id})
        
        return user_options
    return []

def get_recommendations(model, ratings_matrix, user_id, n=5, method='hybrid'):
    """Generate recommendations for a specific user"""
    if model is None or ratings_matrix is None:
        return []
    
    # Check if user exists in the model
    if user_id not in model.user_to_idx:
        st.error(f"User ID '{user_id}' not found in the training data.")
        return []
    
    # Get user index and recommendations
    user_idx = model.user_to_idx[user_id]
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Progress callback for recommendation process
    def progress_callback(step, total, message):
        progress = int(step/total * 100)
        progress_bar.progress(progress)
        status_text.text(message)
    
    # Monkey patch the recommend_top_n method to report progress
    original_recommend = model.recommend_top_n
    
    def recommend_with_progress(ratings_matrix, user_idx, n=10, method='hybrid', min_predicted_rating=3.5):
        # Get all items user hasn't rated
        user_ratings = ratings_matrix[user_idx].toarray().flatten()
        unrated_items = np.where(user_ratings == 0)[0]
        
        # First get baseline predictions
        progress_callback(0, 2, f"Getting baseline predictions for {len(unrated_items)} items...")
        baseline_predictions = []
        for i, item_id in enumerate(unrated_items):
            if i % 1000 == 0:
                progress_callback(0.5 * i/len(unrated_items), 2, f"Processed {i}/{len(unrated_items)} baseline predictions...")
            pred = model.predict(ratings_matrix, user_idx, item_id, 'baseline')
            if pred >= min_predicted_rating:
                baseline_predictions.append((item_id, pred))
        
        # Sort and take top candidates
        baseline_predictions.sort(key=lambda x: x[1], reverse=True)
        candidates = [item_id for item_id, _ in baseline_predictions[:min(len(baseline_predictions), n*3)]]
        
        progress_callback(1, 2, f"Computing detailed predictions for {len(candidates)} candidate items...")
        
        # Get detailed predictions
        predictions = []
        for i, item_id in enumerate(candidates):
            progress_callback(1 + 0.5 * i/len(candidates), 2, f"Processing candidate {i+1}/{len(candidates)}...")
            pred = model.predict(ratings_matrix, user_idx, item_id, method)
            if pred >= min_predicted_rating:
                predictions.append((item_id, pred))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        progress_callback(2, 2, "Recommendations complete!")
        
        # Return top N
        return predictions[:n]
    
    # Replace the method temporarily
    model.recommend_top_n = recommend_with_progress
    
    try:
        # Get recommendations
        recommended_items = model.recommend_top_n(ratings_matrix, user_idx, n=n, method=method)
    finally:
        # Restore the original method
        model.recommend_top_n = original_recommend
    
    return recommended_items

def get_user_profile(model, ratings_matrix, user_id, business_info):
    """Get the user's rating profile"""
    if user_id not in model.user_to_idx:
        return None
    
    user_idx = model.user_to_idx[user_id]
    user_ratings = ratings_matrix[user_idx].toarray().flatten()
    rated_items = np.where(user_ratings > 0)[0]
    
    profile = []
    for item_idx in rated_items:
        business_id = model.idx_to_business[item_idx]
        rating = user_ratings[item_idx]
        
        business_name = business_id
        if business_id in business_info:
            business_name = business_info[business_id].get('name', business_id)
        
        profile.append({
            'business_id': business_id,
            'business_name': business_name,
            'rating': rating
        })
    
    # Sort by rating (highest first)
    profile.sort(key=lambda x: x['rating'], reverse=True)
    return profile

def display_recommendations(recommendations, model, business_info):
    """Display the recommendations in a nice format"""
    if not recommendations:
        st.warning("No recommendations found.")
        return
    
    st.subheader("Top Recommendations")
    
    # Extract categories for word cloud
    all_categories = []
    
    for i, (item_idx, predicted_rating) in enumerate(recommendations):
        business_id = model.idx_to_business[item_idx]
        
        # Create a card-like display for each recommendation
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Display rating as a big number
            st.markdown(f"<h1 style='text-align: center; color: #FF9903;'>{predicted_rating:.1f}⭐</h1>", 
                        unsafe_allow_html=True)
        
        with col2:
            # Get business details if available
            if business_id in business_info:
                business = business_info[business_id]
                name = business.get('name', 'Unknown Business')
                city = business.get('city', 'Unknown')
                state = business.get('state', '')
                categories = business.get('categories', '')
                
                if isinstance(categories, str) and categories.startswith('['):
                    try:
                        # Extract categories for word cloud
                        category_list = json.loads(categories)
                        all_categories.extend(category_list)
                        categories = ', '.join(category_list)
                    except:
                        pass
                
                st.markdown(f"### {name}")
                st.markdown(f"**Business ID:** {business_id}")
                st.markdown(f"**Location:** {city}, {state}")
                if categories:
                    st.markdown(f"**Categories:** {categories}")
            else:
                st.markdown(f"### Business ID: {business_id}")
                st.markdown("*No additional business information available*")
        
        st.markdown("---")
    
    # Create word cloud from categories if available
    if all_categories:
        st.subheader("Recommended Business Categories")
        
        # Count frequencies
        category_freq = {}
        for cat in all_categories:
            if cat in category_freq:
                category_freq[cat] += 1
            else:
                category_freq[cat] = 1
        
        # Generate word cloud
        wc = WordCloud(width=800, height=400, background_color='white', 
                      colormap='viridis', max_words=100)
        wc.generate_from_frequencies(category_freq)
        
        # Display word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout()
        st.pyplot(plt)

def display_user_profile(user_profile):
    """Display the user's rating profile"""
    if not user_profile:
        st.info("No rating history available for this user.")
        return
    
    st.subheader("User Rating History")
    
    # Create bar chart of ratings
    chart_data = pd.DataFrame(user_profile)
    chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('business_name:N', title='Business', sort='-y', axis=alt.Axis(labels=False)),
        y=alt.Y('rating:Q', title='Rating', scale=alt.Scale(domain=[0, 5])),
        color=alt.Color('rating:Q', scale=alt.Scale(scheme='yelloworangered'), legend=None),
        tooltip=['business_name', 'rating']
    ).properties(
        width=600,
        height=300,
        title='Rating Distribution'
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    
    # Display rating distribution
    ratings_count = pd.DataFrame(user_profile)['rating'].value_counts().sort_index()
    
    # Create a DataFrame with all possible ratings (1-5)
    all_ratings = pd.DataFrame({'rating': range(1, 6)})
    # Merge with actual counts, filling missing values with 0
    ratings_dist = all_ratings.merge(ratings_count.reset_index().rename(columns={'index': 'rating', 0: 'count'}), 
                                  on='rating', how='left').fillna(0)
    
    # Create a pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(ratings_dist['count'], labels=[f"{r} ★" for r in ratings_dist['rating']], 
           autopct='%1.1f%%', startangle=90, colors=plt.cm.YlOrRd(np.linspace(0.2, 0.8, 5)))
    ax.axis('equal')
    plt.title('Rating Distribution')
    st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Rating", f"{np.mean([p['rating'] for p in user_profile]):.2f} ★")
    with col2:
        st.metric("Number of Ratings", len(user_profile))
    
    # Top rated businesses
    st.subheader("Top Rated Businesses")
    top_rated = [p for p in user_profile if p['rating'] >= 4]
    if top_rated:
        for p in top_rated[:5]:  # Show top 5
            st.markdown(f"- **{p['business_name']}**: {p['rating']} ★")
    else:
        st.info("No highly rated businesses found.")

def display_model_metrics():
    """Display model metrics from the output"""
    metrics = {
        'RMSE': {
            'user': 1.183,
            'item': 1.157,
            'hybrid': 1.101,
            'baseline': 1.020
        },
        'MAE': {
            'user': 0.902,
            'item': 0.884,
            'hybrid': 0.849,
            'baseline': 0.794
        },
        'Precision@4': {
            'user': 0.768,
            'item': 0.792,
            'hybrid': 0.802,
            'baseline': 0.843
        },
        'Recall@4': {
            'user': 0.474,
            'item': 0.491,
            'hybrid': 0.454,
            'baseline': 0.338
        },
        'F1@4': {
            'user': 0.586,
            'item': 0.606,
            'hybrid': 0.580,
            'baseline': 0.483
        }
    }
    
    st.subheader("Model Performance Metrics")
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Charts", "Detailed Metrics"])
    
    with tab1:
        # Plot the metrics
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        methods = ['user', 'item', 'hybrid', 'baseline']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Error metrics (lower is better)
        axes[0].bar(methods, [metrics['RMSE'][method] for method in methods], color=colors)
        axes[0].set_title('RMSE (Lower is Better)')
        axes[0].set_ylabel('RMSE')
        axes[0].set_ylim(0, 1.5)
        
        # Precision and Recall (higher is better)
        bottom = np.zeros(len(methods))
        for i, metric in enumerate(['Precision@4', 'Recall@4']):
            values = [metrics[metric][method] for method in methods]
            axes[1].bar([f"{method}" for method in methods], values, label=metric, bottom=bottom if i > 0 else None)
            if i > 0:
                bottom += values
        axes[1].set_title('Precision and Recall (Higher is Better)')
        axes[1].set_ylim(0, 1.5)
        axes[1].legend()
        
        # F1 Score (higher is better)
        axes[2].bar(methods, [metrics['F1@4'][method] for method in methods], color=colors)
        axes[2].set_title('F1 Score (Higher is Better)')
        axes[2].set_ylim(0, 1.0)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        # Show detailed metrics in tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Error Metrics (Lower is Better)")
            error_df = pd.DataFrame({
                'Method': methods,
                'RMSE': [metrics['RMSE'][method] for method in methods],
                'MAE': [metrics['MAE'][method] for method in methods]
            })
            st.dataframe(error_df, use_container_width=True)
        
        with col2:
            st.subheader("Recommendation Quality (Higher is Better)")
            quality_df = pd.DataFrame({
                'Method': methods,
                'Precision@4': [metrics['Precision@4'][method] for method in methods],
                'Recall@4': [metrics['Recall@4'][method] for method in methods],
                'F1@4': [metrics['F1@4'][method] for method in methods]
            })
            st.dataframe(quality_df, use_container_width=True)
        
        # Add explanations
        st.markdown("""
        ### Metric Explanations
        
        - **RMSE (Root Mean Square Error)**: Measures the average magnitude of prediction errors. Lower values indicate better performance.
        - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual ratings. Lower values are better.
        - **Precision@4**: Proportion of recommended items with actual rating ≥ 4 out of all recommended items predicted to have rating ≥ 4.
        - **Recall@4**: Proportion of items with actual rating ≥ 4 that were correctly recommended with predicted rating ≥ 4.
        - **F1@4**: Harmonic mean of Precision@4 and Recall@4, balancing both metrics.
        """)

def recommend_similar_users(model, user_id, n=5):
    """Find similar users based on similarity matrix"""
    if model is None or model.user_similarity is None:
        return []
    
    if user_id not in model.user_to_idx:
        return []
    
    user_idx = model.user_to_idx[user_id]
    
    # Get similarity scores for this user
    similarities = model.user_similarity[user_idx].toarray().flatten()
    
    # Get top similar users (excluding self)
    similar_users = []
    for idx in np.argsort(similarities)[::-1]:
        if idx != user_idx and similarities[idx] > 0:
            similar_user_id = model.idx_to_user[idx]
            similar_users.append((similar_user_id, similarities[idx]))
            if len(similar_users) >= n:
                break
    
    return similar_users

def main():
    st.set_page_config(
        page_title="Yelp Recommendation System",
        page_icon="🍽️",
        layout="wide"
    )
    
    st.title("Yelp Business Recommendation System")
    st.markdown("""
    This app demonstrates a collaborative filtering recommendation system for Yelp businesses.
    Select a user to get personalized business recommendations based on their preferences.
    """)
    
    # Load model and data
    model, ratings_matrix = load_model_and_data()
    
    if model is None:
        st.error("Could not load the model. Please make sure the model directory exists.")
        st.stop()
    
    # Load business and user information if available
    business_info = load_business_data()
    user_info = load_user_data()
    
    # Initialize session state for storing user selection
    if 'selected_user_id' not in st.session_state:
        st.session_state.selected_user_id = None
    
    # Add module reloading button to sidebar
    with st.sidebar:
        if st.button("Reload Modules"):
            import importlib
            import utils  # Assuming this is your module with YelpRecommendationSystem
            importlib.reload(utils)
            st.success("Backend modules reloaded!")
            
    # Display sidebar for user selection and parameters
    with st.sidebar:
        st.header("Settings")
        
        # Option to enter a specific user ID
        user_id_input = st.text_input("Enter a specific User ID (if known)")
        
        # Show dropdown of sample user IDs with names
        user_options = get_user_ids_with_names(model, user_info, n=100)
        
        # Format for display
        display_options = [u["label"] for u in user_options]
        user_values = [u["value"] for u in user_options]
        
        selected_option = st.selectbox(
            "Or select a user from the sample list",
            options=display_options,
            index=0 if display_options else None
        )
        
        # Map selected option back to user ID
        if selected_option:
            selected_idx = display_options.index(selected_option)
            selected_user_id = user_values[selected_idx]
            # Store in session state
            st.session_state.selected_user_id = selected_user_id
        else:
            selected_user_id = None
        
        # Use the entered ID if provided, otherwise use the selected one
        user_id = user_id_input if user_id_input else st.session_state.selected_user_id
        
        # Method selection
        method = st.radio(
            "Recommendation Method",
            ["hybrid", "user", "item", "baseline"],
            index=0,
            help="The algorithm used to generate recommendations"
        )
        
        # Number of recommendations
        num_recommendations = st.slider(
            "Number of Recommendations",
            min_value=1,
            max_value=20,
            value=5
        )
        
        # Display options
        st.subheader("Display Options")
        show_profile = st.checkbox("Show User Profile", value=True)
        show_similar_users = st.checkbox("Show Similar Users", value=True)
        show_metrics = st.checkbox("Show Model Metrics", value=False)
        
        # Get recommendations button - explicitly log the user ID being used
        get_recs_button = st.button("Get Recommendations", type="primary")
        
        # Debug output - show which user ID is currently selected
        st.write(f"Current User ID: {user_id}")
    
    # Main page content
    if not user_id:
        st.info("Please select a user ID from the sidebar or enter a specific one.")
    else:
        # Extract user name from selection or create placeholder
        user_name = "Unknown User"
        if user_id in user_info and 'name' in user_info[user_id]:
            user_name = user_info[user_id]['name']
        else:
            # Find user name from format string
            for option in user_options:
                if option["value"] == user_id:
                    # Extract name from "Name (user_id)" format
                    parts = option["label"].split(" (")
                    if len(parts) > 1:
                        user_name = parts[0]
                    break
        
        # Display user information
        st.header(f"Recommendations for {user_name}")
        st.subheader(f"User ID: {user_id}")
        
        # Create tabs for different sections
        tabs = []
        tabs.append("Recommendations")
        if show_profile:
            tabs.append("User Profile")
        if show_similar_users:
            tabs.append("Similar Users")
        if show_metrics:
            tabs.append("Model Metrics")
        
        tab_selection = st.tabs(tabs)
        
        # Recommendations tab
        with tab_selection[0]:
            # Get and display recommendations when button is pressed
            if get_recs_button:
                with st.spinner(f"Generating {num_recommendations} recommendations using {method} method..."):
                    recommendations = get_recommendations(
                        model, 
                        ratings_matrix, 
                        user_id, 
                        n=num_recommendations, 
                        method=method
                    )
                    display_recommendations(recommendations, model, business_info)
            else:
                st.info("Click 'Get Recommendations' to generate personalized recommendations")
        
        # User Profile tab
        if show_profile:
            tab_index = tabs.index("User Profile")
            with tab_selection[tab_index]:
                user_profile = get_user_profile(model, ratings_matrix, user_id, business_info)
                display_user_profile(user_profile)
        
        # Similar Users tab
        if show_similar_users:
            tab_index = tabs.index("Similar Users")
            with tab_selection[tab_index]:
                similar_users = recommend_similar_users(model, user_id, n=10)
                
                if similar_users:
                    st.subheader("Users with Similar Preferences")
                    
                    # Display similar users as a table
                    similar_data = []
                    for sim_user_id, similarity in similar_users:
                        # Get user name if available
                        sim_user_name = "Unknown User"
                        if sim_user_id in user_info and 'name' in user_info[sim_user_id]:
                            sim_user_name = user_info[sim_user_id]['name']
                        else:
                            # Create a placeholder name from user ID
                            seed = hash(sim_user_id) % 10000
                            random.seed(seed)
                            first = random.choice(["Alex", "Jordan", "Taylor", "Morgan", "Casey"])
                            last = random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones"])
                            sim_user_name = f"{first} {last}"
                        
                        similar_data.append({
                            "User Name": sim_user_name,
                            "User ID": sim_user_id,
                            "Similarity Score": f"{similarity:.3f}"
                        })
                    
                    st.dataframe(pd.DataFrame(similar_data), use_container_width=True)
                    
                    # Add explanation
                    st.info("""
                    These users have similar rating patterns. The similarity score represents how closely their preferences
                    match the selected user's preferences. Users with higher scores are more likely to enjoy the same businesses.
                    """)
                else:
                    st.info("No similar users found.")
        
        # Model Metrics tab
        if show_metrics:
            tab_index = tabs.index("Model Metrics")
            with tab_selection[tab_index]:
                display_model_metrics()
    
    # Add information about the model in footer
    st.sidebar.markdown("---")
    st.sidebar.subheader("About the Model")
    st.sidebar.markdown("""
    This recommendation system uses collaborative filtering with the following methods:
    
    - **User-based**: Recommends based on similar users' preferences
    - **Item-based**: Recommends items similar to those the user liked
    - **Hybrid**: Combines both user and item approaches
    - **Baseline**: Simple prediction using average ratings
    """)
    
    # Add footer
    st.markdown("---")
    st.caption("Yelp Recommendation System | Built with Streamlit")

if __name__ == "__main__":
    main()