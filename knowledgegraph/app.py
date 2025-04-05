import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import joblib
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from neo4j import GraphDatabase
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

class YelpRecommenderApp:
    def __init__(self, uri=None, user=None, password=None, models_folder='models'):
        self.uri = uri
        self.user = user
        self.password = password
        self.models_folder = models_folder
        self.driver = None
        
        # Load models and mappings
        self.load_models_and_mappings()
        
    def connect_to_neo4j(self):
        """Connect to Neo4j database"""
        if not self.uri or not self.user or not self.password:
            return False
            
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    return True
                return False
        except Exception as e:
            st.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    def close_neo4j_connection(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.driver = None
    
    def load_models_and_mappings(self):
        """Load saved models and mappings"""
        # Initialize attributes
        self.user_to_idx = {}
        self.business_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_business = {}
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.hybrid_weights = None
        self.graph_queries = None
        self.evaluation_metrics = None
        
        # Try to load mappings
        try:
            mappings_path = os.path.join(self.models_folder, 'mappings')
            if os.path.exists(mappings_path):
                with open(os.path.join(mappings_path, 'user_to_idx.pkl'), 'rb') as f:
                    self.user_to_idx = pickle.load(f)
                
                with open(os.path.join(mappings_path, 'business_to_idx.pkl'), 'rb') as f:
                    self.business_to_idx = pickle.load(f)
                
                with open(os.path.join(mappings_path, 'idx_to_user.pkl'), 'rb') as f:
                    self.idx_to_user = pickle.load(f)
                
                with open(os.path.join(mappings_path, 'idx_to_business.pkl'), 'rb') as f:
                    self.idx_to_business = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load mappings: {e}")
        
        # Try to load User-CF model
        try:
            user_cf_path = os.path.join(self.models_folder, 'user_cf_model')
            if os.path.exists(user_cf_path):
                self.user_item_matrix = pd.read_pickle(os.path.join(user_cf_path, 'user_item_matrix.pkl'))
                self.user_similarity = joblib.load(os.path.join(user_cf_path, 'user_similarity.pkl'))
        except Exception as e:
            st.warning(f"Could not load User-CF model: {e}")
        
        # Try to load Item-CF model
        try:
            item_cf_path = os.path.join(self.models_folder, 'item_cf_model')
            if os.path.exists(item_cf_path):
                # Use the same matrix if already loaded from user-cf
                if self.user_item_matrix is None:
                    self.user_item_matrix = pd.read_pickle(os.path.join(item_cf_path, 'user_item_matrix.pkl'))
                self.item_similarity = joblib.load(os.path.join(item_cf_path, 'item_similarity.pkl'))
        except Exception as e:
            st.warning(f"Could not load Item-CF model: {e}")
        
        # Load hybrid model weights
        try:
            hybrid_path = os.path.join(self.models_folder, 'hybrid_model')
            if os.path.exists(hybrid_path):
                with open(os.path.join(hybrid_path, 'weights.pkl'), 'rb') as f:
                    self.hybrid_weights = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load hybrid model weights: {e}")
            self.hybrid_weights = {'alpha': 0.7, 'beta': 0.2, 'gamma': 0.1}  # Default weights
        
        # Load graph model queries
        try:
            graph_path = os.path.join(self.models_folder, 'graph_model')
            if os.path.exists(graph_path):
                with open(os.path.join(graph_path, 'queries.pkl'), 'rb') as f:
                    self.graph_queries = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load graph model queries: {e}")
        
        # Load evaluation metrics
        try:
            eval_path = os.path.join(self.models_folder, 'evaluation')
            if os.path.exists(eval_path):
                with open(os.path.join(eval_path, 'metrics.pkl'), 'rb') as f:
                    self.evaluation_metrics = pickle.load(f)
                
                if os.path.exists(os.path.join(eval_path, 'metrics_table.csv')):
                    self.metrics_table = pd.read_csv(os.path.join(eval_path, 'metrics_table.csv'), index_col=0)
        except Exception as e:
            st.warning(f"Could not load evaluation metrics: {e}")
    
    def fetch_user_details(self, user_id):
        """Fetch user details from Neo4j"""
        if not self.driver:
            return None
            
        try:
            with self.driver.session() as session:
                result = session.run("""
                MATCH (u:User {user_id: $user_id})
                RETURN u.name as name, 
                       u.review_count as review_count,
                       u.average_stars as average_stars,
                       u.yelping_since as yelping_since
                """, user_id=user_id)
                
                record = result.single()
                if record:
                    return dict(record)
                return None
        except Exception as e:
            st.error(f"Error fetching user details: {e}")
            return None
    
    def fetch_user_reviews(self, user_id, limit=5):
        """Fetch user's past reviews from Neo4j"""
        if not self.driver:
            return []
            
        try:
            with self.driver.session() as session:
                result = session.run("""
                MATCH (u:User {user_id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
                RETURN b.name as business_name, 
                       b.business_id as business_id,
                       r.stars as stars,
                       r.date as date,
                       r.text as review_text
                ORDER BY r.date DESC
                LIMIT $limit
                """, user_id=user_id, limit=limit)
                
                return [dict(record) for record in result]
        except Exception as e:
            st.error(f"Error fetching user reviews: {e}")
            return []
    
    def fetch_business_details(self, business_id):
        """Fetch business details from Neo4j"""
        if not self.driver:
            return None
            
        try:
            with self.driver.session() as session:
                result = session.run("""
                MATCH (b:Business {business_id: $business_id})
                OPTIONAL MATCH (b)-[:IN_CATEGORY]->(c:Category)
                RETURN b.name as name, 
                       b.stars as stars,
                       b.review_count as review_count,
                       b.city as city,
                       b.state as state,
                       b.address as address,
                       b.postal_code as postal_code,
                       b.latitude as latitude,
                       b.longitude as longitude,
                       collect(distinct c.name) as categories
                """, business_id=business_id)
                
                record = result.single()
                if record:
                    return dict(record)
                return None
        except Exception as e:
            st.error(f"Error fetching business details: {e}")
            return None
    
    def get_user_cf_recommendations(self, user_id, k=10, n=10):
        """Get user-based collaborative filtering recommendations"""
        if not self.user_item_matrix is not None or self.user_similarity is None:
            return []
        
        # Convert user_id to user_idx
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # Check if user is in the similarity matrix
        if user_idx >= len(self.user_similarity):
            return []
        
        # Get similarity scores for this user
        sim_scores = self.user_similarity[user_idx]
        
        # Get top-k similar users (excluding self)
        similar_users = np.argsort(sim_scores)[::-1][1:k+1]
        
        # Get items rated by the target user
        user_rated_items = set(self.user_item_matrix.columns[self.user_item_matrix.loc[user_idx] > 0])
        
        # Get recommendations from similar users
        candidate_items = {}
        
        for sim_user in similar_users:
            # Weight of this similar user (similarity score)
            weight = sim_scores[sim_user]
            
            # Items rated highly by this similar user
            sim_user_ratings = self.user_item_matrix.loc[sim_user]
            rated_items = sim_user_ratings[sim_user_ratings > 0].index
            
            for item in rated_items:
                rating = sim_user_ratings[item]
                
                # Skip items already rated by target user
                if item in user_rated_items:
                    continue
                
                # Weighted rating
                if item not in candidate_items:
                    candidate_items[item] = {'weighted_sum': 0, 'similarity_sum': 0}
                
                candidate_items[item]['weighted_sum'] += weight * rating
                candidate_items[item]['similarity_sum'] += weight
        
        # Calculate predicted ratings
        recommendations = []
        for item, values in candidate_items.items():
            if values['similarity_sum'] > 0:
                predicted_rating = values['weighted_sum'] / values['similarity_sum']
                
                # Convert item_idx back to business_id
                business_id = self.idx_to_business.get(item)
                if business_id:
                    recommendations.append({
                        'business_id': business_id,
                        'predicted_rating': predicted_rating,
                        'score': predicted_rating  # Use predicted rating as score
                    })
        
        # Sort recommendations by predicted rating
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        
        # Get top-n recommendations
        return recommendations[:n]
    
    def get_item_cf_recommendations(self, user_id, k=10, n=10):
        """Get item-based collaborative filtering recommendations"""
        if self.user_item_matrix is None or self.item_similarity is None:
            return []
        
        # Convert user_id to user_idx
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # Check if user is in the matrix
        if user_idx not in self.user_item_matrix.index:
            return []
        
        # Get items rated by this user
        user_ratings = self.user_item_matrix.loc[user_idx]
        user_rated_items = user_ratings[user_ratings > 0].index
        
        if len(user_rated_items) == 0:
            return []
        
        # Calculate predicted ratings for unrated items
        candidate_items = {}
        
        for item in user_rated_items:
            rating = user_ratings[item]
            
            # Check if item is in the similarity matrix
            if item >= len(self.item_similarity):
                continue
            
            # Get similar items
            sim_scores = self.item_similarity[item]
            
            # Get items not rated by the user
            all_items = set(range(len(sim_scores)))
            unrated_items = all_items - set(user_rated_items)
            
            for unrated_item in unrated_items:
                if unrated_item >= len(sim_scores):
                    continue
                
                similarity = sim_scores[unrated_item]
                
                if unrated_item not in candidate_items:
                    candidate_items[unrated_item] = {'weighted_sum': 0, 'similarity_sum': 0}
                
                candidate_items[unrated_item]['weighted_sum'] += similarity * rating
                candidate_items[unrated_item]['similarity_sum'] += similarity
        
        # Calculate predicted ratings
        recommendations = []
        for item, values in candidate_items.items():
            if values['similarity_sum'] > 0:
                predicted_rating = values['weighted_sum'] / values['similarity_sum']
                
                # Convert item_idx back to business_id
                business_id = self.idx_to_business.get(item)
                if business_id:
                    recommendations.append({
                        'business_id': business_id,
                        'predicted_rating': predicted_rating,
                        'score': predicted_rating  # Use predicted rating as score
                    })
        
        # Sort recommendations by predicted rating
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        
        # Get top-n recommendations
        return recommendations[:n]
    
    def get_graph_recommendations(self, user_id, max_items=10):
        """Get graph-based recommendations using Neo4j"""
        if not self.driver:
            return []
        
        try:
            recommendations = []
            
            with self.driver.session() as session:
                # Get recommendations based on common categories with highly-rated businesses
                result = session.run("""
                MATCH (u:User {user_id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)-[:IN_CATEGORY]->(c:Category)
                WHERE r.stars >= 4
                WITH u, c, count(*) as category_weight
                
                MATCH (c)<-[:IN_CATEGORY]-(rec_business:Business)
                WHERE NOT EXISTS((u)-[:WROTE]->(:Review)-[:ABOUT]->(rec_business))
                
                WITH rec_business, sum(category_weight) as score, collect(distinct c.name) as matched_categories
                ORDER BY score DESC, rec_business.stars DESC
                LIMIT $max_items
                
                RETURN rec_business.business_id as business_id, 
                        rec_business.name as name,
                        rec_business.stars as avg_rating,
                        score,
                        matched_categories
                """, user_id=user_id, max_items=max_items)
                
                recommendations = [dict(record) for record in result]
                
                # If not enough category-based recommendations, supplement with collaborative approach
                if len(recommendations) < max_items:
                    collab_result = session.run("""
                    MATCH (u1:User {user_id: $user_id})-[:WROTE]->(r1:Review)-[:ABOUT]->(b:Business)
                    <-[:ABOUT]-(r2:Review)<-[:WROTE]-(u2:User)
                    WHERE r1.stars >= 4 AND r2.stars >= 4 AND u1 <> u2
                    
                    WITH u1, u2, count(distinct b) as common_likes
                    ORDER BY common_likes DESC
                    LIMIT 10
                    
                    MATCH (u2)-[:WROTE]->(r:Review)-[:ABOUT]->(rec_business:Business)
                    WHERE r.stars >= 4
                    AND NOT EXISTS((u1)-[:WROTE]->(:Review)-[:ABOUT]->(rec_business))
                    
                    WITH rec_business, sum(common_likes) as score
                    ORDER BY score DESC, rec_business.stars DESC
                    LIMIT $remaining
                    
                    RETURN rec_business.business_id as business_id,
                            rec_business.name as name,
                            rec_business.stars as avg_rating,
                            score,
                            [] as matched_categories
                    """, user_id=user_id, remaining=max_items-len(recommendations))
                    
                    collab_recs = [dict(record) for record in collab_result]
                    recommendations.extend(collab_recs)
            
            return recommendations
        except Exception as e:
            st.error(f"Error getting graph recommendations: {e}")
            return []
    
    def get_hybrid_recommendations(self, user_id, max_items=10):
        """Get hybrid recommendations combining all methods"""
        # Get recommendations from different approaches
        user_cf_recs = self.get_user_cf_recommendations(user_id, n=50)
        item_cf_recs = self.get_item_cf_recommendations(user_id, n=50)
        graph_recs = self.get_graph_recommendations(user_id, max_items=50)
        
        # Get weights
        if self.hybrid_weights:
            alpha = self.hybrid_weights.get('alpha', 0.7)
            beta = self.hybrid_weights.get('beta', 0.2)
            gamma = self.hybrid_weights.get('gamma', 0.1)
        else:
            alpha, beta, gamma = 0.7, 0.2, 0.1
        
        # Combine recommendations
        business_scores = defaultdict(float)
        
        # Add user-based CF scores
        for rec in user_cf_recs:
            business_id = rec['business_id']
            score = rec['score']
            business_scores[business_id] += alpha * score
        
        # Add item-based CF scores
        for rec in item_cf_recs:
            business_id = rec['business_id']
            score = rec['score']
            business_scores[business_id] += beta * score
        
        # Add graph-based scores
        for rec in graph_recs:
            business_id = rec['business_id']
            score = min(5, rec['avg_rating'])  # Normalize score to be between 0-5
            business_scores[business_id] += gamma * score
        
        # Sort businesses by score
        sorted_businesses = sorted(
            business_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get top recommendations
        top_businesses = sorted_businesses[:max_items]
        
        # Fetch details for these businesses
        recommendations = []
        for business_id, score in top_businesses:
            recommendations.append({
                'business_id': business_id,
                'score': score
            })
        
        return recommendations

    def fetch_all_users_sample(self, limit=1000):
        """Fetch a sample of users from Neo4j"""
        if not self.driver:
            return []
            
        try:
            with self.driver.session() as session:
                result = session.run("""
                MATCH (u:User)-[:WROTE]->(:Review)
                RETURN u.user_id as user_id, u.name as name, u.review_count as review_count
                ORDER BY u.review_count DESC
                LIMIT $limit
                """, limit=limit)
                
                return [dict(record) for record in result]
        except Exception as e:
            st.error(f"Error fetching users: {e}")
            return []

def main():
    st.set_page_config(
        page_title="Yelp Recommender System",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🍽️ Yelp Recommender System")
    
    # Initialize app state
    if 'app' not in st.session_state:
        st.session_state.app = YelpRecommenderApp()
    
    app = st.session_state.app
    
    # Sidebar - Neo4j Connection
    with st.sidebar:
        st.header("Neo4j Connection")
        
        # Connection form
        with st.form("neo4j_connection"):
            uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
            user = st.text_input("Username", value="neo4j")
            password = st.text_input("Password", type="password", value="password")
            
            connect_btn = st.form_submit_button("Connect")
            
            if connect_btn:
                app.uri = uri
                app.user = user
                app.password = password
                
                if app.connect_to_neo4j():
                    st.success("Connected to Neo4j!")
                else:
                    st.error("Failed to connect to Neo4j")
        
        # Check connection status
        if app.driver:
            st.success("✅ Connected to Neo4j")
        else:
            st.warning("❌ Not connected to Neo4j")
        
        st.divider()
        
        # Navigation
        st.header("Navigation")
        page = st.radio("Go to", ["Dashboard", "User Recommendations", "Model Evaluation"])
    
    # Main content
    if page == "Dashboard":
        show_dashboard(app)
    elif page == "User Recommendations":
        show_user_recommendations(app)
    else:  # Model Evaluation
        show_model_evaluation(app)

def show_dashboard(app):
    st.header("System Dashboard")
    
    # Status of loaded models
    st.subheader("Model Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if app.user_item_matrix is not None and app.user_similarity is not None:
            st.success("✅ User-CF Model Loaded")
        else:
            st.error("❌ User-CF Model Not Loaded")
    
    with col2:
        if app.user_item_matrix is not None and app.item_similarity is not None:
            st.success("✅ Item-CF Model Loaded")
        else:
            st.error("❌ Item-CF Model Not Loaded")
    
    with col3:
        if app.graph_queries is not None:
            st.success("✅ Graph Model Loaded")
        else:
            st.error("❌ Graph Model Not Loaded")
    
    # Display model statistics
    st.subheader("Model Statistics")
    
    # User and business counts
    if app.user_to_idx and app.business_to_idx:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Users", f"{len(app.user_to_idx):,}")
        with col2:
            st.metric("Total Businesses", f"{len(app.business_to_idx):,}")
    
    # User-Item Matrix Statistics
    if app.user_item_matrix is not None:
        st.subheader("User-Item Matrix")
        
        # Calculate sparsity
        total_elements = app.user_item_matrix.shape[0] * app.user_item_matrix.shape[1]
        non_zero = app.user_item_matrix.values.nonzero()[0].size
        sparsity = 100 * (1 - non_zero / total_elements)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Matrix Shape", f"{app.user_item_matrix.shape[0]} × {app.user_item_matrix.shape[1]}")
        with col2:
            st.metric("Non-zero Elements", f"{non_zero:,}")
        with col3:
            st.metric("Sparsity", f"{sparsity:.2f}%")
    
    # Hybrid model weights
    if app.hybrid_weights:
        st.subheader("Hybrid Model Weights")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("User-CF (α)", f"{app.hybrid_weights.get('alpha', 0.7):.2f}")
        with col2:
            st.metric("Item-CF (β)", f"{app.hybrid_weights.get('beta', 0.2):.2f}")
        with col3:
            st.metric("Graph-based (γ)", f"{app.hybrid_weights.get('gamma', 0.1):.2f}")
    
    # Quick visualization of model performance
    if app.evaluation_metrics:
        st.subheader("Model Performance Overview")
        
        metrics_data = app.metrics_table.copy() if hasattr(app, 'metrics_table') else None
        
        if metrics_data is not None:
            # Replace 'N/A' strings with NaN
            metrics_data = metrics_data.replace('N/A', np.nan)
            
            # Convert to numeric
            for col in metrics_data.columns:
                metrics_data[col] = pd.to_numeric(metrics_data[col], errors='coerce')
            
            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot F1 score, precision, and recall
            metrics_to_plot = ['precision@N', 'recall@N', 'f1@N', 'ndcg@N']
            plot_data = metrics_data[metrics_to_plot].copy()
            
            # Check if we have any valid data
            if not plot_data.isnull().all().all():
                plot_data.plot(kind='bar', ax=ax)
                plt.title('Recommendation Performance Metrics')
                plt.ylabel('Score')
                plt.xlabel('Model')
                plt.ylim(0, 1.0)
                plt.legend(title='Metric')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                st.pyplot(fig)
            else:
                st.warning("No valid performance metrics available to plot")
        else:
            st.warning("Performance metrics not available")

def show_user_recommendations(app):
    st.header("User Recommendations")
    
    # Check if connected to Neo4j
    if not app.driver:
        st.warning("Please connect to Neo4j in the sidebar to access user recommendations")
        return
    
    # User selection
    st.subheader("Select a User")
    
    # Option to search by user ID
    custom_user_id = st.text_input("Enter User ID (or select from dropdown below)")
    
    # Or select from a list of users
    users = app.fetch_all_users_sample(limit=100)
    
    if not users:
        st.warning("No users found or database connection issue")
        return
    
    user_options = [f"{user['name']} ({user['user_id']}) - {user['review_count']} reviews" for user in users]
    selected_user_option = st.selectbox("Select a user", [""] + user_options)
    
    # Extract user_id from selection
    user_id = None
    if custom_user_id:
        user_id = custom_user_id
    elif selected_user_option:
        # Extract user_id from the format "Name (user_id) - X reviews"
        import re
        match = re.search(r'\((.*?)\)', selected_user_option)
        if match:
            user_id = match.group(1)
    
    if not user_id:
        st.info("Please select a user to view recommendations")
        return
    
    # Get user details
    user_details = app.fetch_user_details(user_id)
    
    if not user_details:
        st.error(f"User with ID '{user_id}' not found")
        return
    
    # Display user info
    st.subheader(f"User: {user_details.get('name', 'Unknown')}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Review Count", user_details.get('review_count', 'N/A'))
    with col2:
        st.metric("Average Rating Given", f"{user_details.get('average_stars', 'N/A'):.1f} ⭐")
    with col3:
        st.metric("Yelping Since", user_details.get('yelping_since', 'N/A'))
    
    # Show user's recent reviews
    st.subheader("Recent Reviews")
    user_reviews = app.fetch_user_reviews(user_id, limit=5)
    
    if user_reviews:
        for i, review in enumerate(user_reviews):
            with st.expander(f"{review['business_name']} - {review['stars']} ⭐ ({review['date']})"):
                st.write(review['review_text'])
    else:
        st.info("No reviews found for this user")
    
    # Recommendation options
    st.subheader("Get Recommendations")
    
    # Select recommendation method
    rec_method = st.radio(
        "Choose recommendation method",
        ["User-based CF", "Item-based CF", "Graph-based", "Hybrid (All methods)"]
    )
    
    num_recs = st.slider("Number of recommendations", 5, 20, 10)
    
    if st.button("Get Recommendations"):
        with st.spinner("Generating recommendations..."):
            # Get recommendations based on selected method
            recommendations = []
            
            if rec_method == "User-based CF":
                recommendations = app.get_user_cf_recommendations(user_id, n=num_recs)
            elif rec_method == "Item-based CF":
                recommendations = app.get_item_cf_recommendations(user_id, n=num_recs)
            elif rec_method == "Graph-based":
                recommendations = app.get_graph_recommendations(user_id, max_items=num_recs)
            else:  # Hybrid
                recommendations = app.get_hybrid_recommendations(user_id, max_items=num_recs)
            
            # Display recommendations
            if not recommendations:
                st.warning(f"No recommendations found using {rec_method}")
                return
            
            st.subheader(f"Top {len(recommendations)} Recommendations")
            
            for i, rec in enumerate(recommendations):
                business_id = rec['business_id']
                
                # Get business details
                business = app.fetch_business_details(business_id)
                
                if not business:
                    continue
                
                # Create a nice card for each recommendation
                with st.container():
                    col1, col2, col3 = st.columns([1, 3, 1])
                    
                    with col1:
                        # Display recommendation rank
                        st.markdown(f"### #{i+1}")
                    
                    with col2:
                        # Business name and rating
                        st.markdown(f"#### {business['name']}")
                        st.markdown(f"**Rating:** {business['stars']:.1f} ⭐ ({business['review_count']} reviews)")
                        
                        # Location
                        st.markdown(f"**Location:** {business['address']}, {business['city']}, {business['state']} {business['postal_code']}")
                        
                        # Categories
                        categories = business.get('categories', [])
                        if categories:
                            st.markdown(f"**Categories:** {', '.join(categories)}")
                    
                    with col3:
                        # Display score/confidence
                        if 'score' in rec:
                            score = rec['score']
                            st.markdown(f"**Match Score:**")
                            st.progress(min(score / 5.0, 1.0))
                            st.markdown(f"{score:.2f}")
                        
                        # If graph-based, show matched categories
                        if rec_method == "Graph-based" and 'matched_categories' in rec:
                            matched = rec.get('matched_categories', [])
                            if matched:
                                st.markdown(f"**Matched categories:**")
                                st.markdown(", ".join(matched[:3]))
                                if len(matched) > 3:
                                    st.markdown(f"...and {len(matched)-3} more")
                    
                    st.divider()

def show_model_evaluation(app):
    st.header("Model Evaluation")
    
    # Check if we have evaluation metrics
    if not app.evaluation_metrics:
        st.warning("Evaluation metrics not available. Please ensure the models are properly loaded.")
        return
    
    # Display overall metrics
    st.subheader("Performance Metrics")
    
    if hasattr(app, 'metrics_table'):
        # Show metrics table
        st.dataframe(app.metrics_table)
    else:
        # Create a basic table from the metrics dict
        metrics_df = pd.DataFrame({
            'Metric': list(app.evaluation_metrics.keys()),
            'Value': list(app.evaluation_metrics.values())
        })
        st.dataframe(metrics_df)
    
    # Show performance visualizations
    st.subheader("Performance Comparison")
    
    # Prepare data for visualization
    if hasattr(app, 'metrics_table'):
        metrics_data = app.metrics_table.copy()
        
        # Replace 'N/A' strings with NaN
        metrics_data = metrics_data.replace('N/A', np.nan)
        
        # Convert to numeric
        for col in metrics_data.columns:
            metrics_data[col] = pd.to_numeric(metrics_data[col], errors='coerce')
        
        # Plot precision, recall, and F1 scores
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics_to_plot = ['precision@N', 'recall@N', 'f1@N', 'ndcg@N']
        available_metrics = [m for m in metrics_to_plot if m in metrics_data.columns]
        
        if available_metrics:
            plot_data = metrics_data[available_metrics].copy()
            
            if not plot_data.isnull().all().all():
                plot_data.plot(kind='bar', ax=ax)
                plt.title('Recommendation Performance Metrics')
                plt.ylabel('Score')
                plt.xlabel('Model')
                plt.ylim(0, 1.0)
                plt.legend(title='Metric')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                st.pyplot(fig)
            else:
                st.warning("No valid metrics data available to plot")
        else:
            st.warning("Required metrics not found in the data")
    
    # Model comparison visualization
    st.subheader("Model Comparison")
    
    # Check if we have detailed evaluation data for visualization
    try:
        if hasattr(app, 'evaluation_metrics') and isinstance(app.evaluation_metrics, dict):
            # Model labels
            models = ['User-CF', 'Item-CF', 'Graph-based', 'Hybrid']
            
            # Extract metrics for each model (example)
            precision = []
            recall = []
            ndcg = []
            
            for model in models:
                model_key = model.lower().replace('-', '_')
                precision.append(app.evaluation_metrics.get(f'{model_key}_precision', np.nan))
                recall.append(app.evaluation_metrics.get(f'{model_key}_recall', np.nan))
                ndcg.append(app.evaluation_metrics.get(f'{model_key}_ndcg', np.nan))
            
            # Create a DataFrame
            comparison_df = pd.DataFrame({
                'Model': models,
                'Precision': precision,
                'Recall': recall,
                'NDCG': ndcg
            })
            
            # Convert to numeric, replacing NaN with 0
            for col in ['Precision', 'Recall', 'NDCG']:
                comparison_df[col] = pd.to_numeric(comparison_df[col], errors='coerce').fillna(0)
            
            # Plot radar chart using matplotlib
            categories = ['Precision', 'Recall', 'NDCG']
            N = len(categories)
            
            # Create angles for each metric
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            
            # Add lines and labels
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            plt.xticks(angles[:-1], categories)
            
            # Draw model performance
            for i, model in enumerate(models):
                values = comparison_df.loc[i, categories].values.flatten().tolist()
                values += values[:1]
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
                ax.fill(angles, values, alpha=0.1)
            
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not generate model comparison chart: {e}")
    
    # Additional evaluation metrics and insights
    st.subheader("Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Strengths")
        st.markdown("""
        - **User-CF**: Effective when users have similar preferences
        - **Item-CF**: Works well for niche items and personalization
        - **Graph-based**: Captures contextual relationships
        - **Hybrid**: Combines strengths of individual approaches
        """)
    
    with col2:
        st.markdown("### Limitations")
        st.markdown("""
        - **User-CF**: Cold-start problem for new users
        - **Item-CF**: Limited by item similarity calculations
        - **Graph-based**: Depends on quality of graph relationships
        - **Hybrid**: Complexity in tuning optimal weights
        """)

if __name__ == "__main__":
    main()