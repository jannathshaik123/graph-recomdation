import os
import pickle
import joblib
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from sklearn.metrics import mean_squared_error, precision_score, recall_score, ndcg_score
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from collections import defaultdict

class YelpRecommender:
    def __init__(self, uri, user, password, models_folder='models'):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.models_folder = models_folder
        
        # Create models folder if it doesn't exist
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
            print(f"Created models folder: {models_folder}")
        
    def close(self):
        self.driver.close()
    
    def fetch_user_business_ratings(self):
        """
        Fetch all user-business ratings from the database
        Returns a pandas DataFrame with user_id, business_id, and stars
        """
        with self.driver.session() as session:
            result = session.run("""
            MATCH (u:User)-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
            RETURN u.user_id AS user_id, b.business_id AS business_id, 
                   r.stars AS rating, r.date AS date
            ORDER BY r.date
            """)
            
            # Convert to pandas DataFrame
            ratings_df = pd.DataFrame([dict(record) for record in result])
            print(f"Fetched {len(ratings_df)} ratings")
            return ratings_df
            
    def fetch_business_features(self):
        """
        Fetch business features for content-based filtering
        """
        with self.driver.session() as session:
            # Get businesses with their categories
            result = session.run("""
            MATCH (b:Business)-[:IN_CATEGORY]->(c:Category)
            RETURN b.business_id AS business_id, 
                   collect(c.name) AS categories,
                   b.city AS city,
                   b.stars AS avg_stars,
                   b.review_count AS review_count
            """)
            
            businesses = []
            for record in result:
                business = dict(record)
                businesses.append(business)
                
            business_df = pd.DataFrame(businesses)
            print(f"Fetched features for {len(business_df)} businesses")
            return business_df
            
    def preprocess_data(self, ratings_df, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets with improved train/test split
        """
        # Sort by date to ensure train data comes before test data
        ratings_df = ratings_df.sort_values('date')
        
        # Create user and business indices
        unique_users = ratings_df['user_id'].unique()
        unique_businesses = ratings_df['business_id'].unique()
        
        user_to_idx = {user: i for i, user in enumerate(unique_users)}
        business_to_idx = {business: i for i, business in enumerate(unique_businesses)}
        
        # Convert user_id and business_id to indices
        ratings_df['user_idx'] = ratings_df['user_id'].map(user_to_idx)
        ratings_df['business_idx'] = ratings_df['business_id'].map(business_to_idx)
        
        # Filter out users with too few ratings for proper train/test split
        # Count ratings per user
        user_rating_counts = ratings_df['user_idx'].value_counts()
        
        # Only keep users with at least 5 ratings to ensure they have data in both train and test
        min_ratings = 5
        users_with_enough_ratings = user_rating_counts[user_rating_counts >= min_ratings].index
        print(f"Filtering from {len(user_rating_counts)} users to {len(users_with_enough_ratings)} users with at least {min_ratings} ratings")
        
        filtered_df = ratings_df[ratings_df['user_idx'].isin(users_with_enough_ratings)]
        
        # Split data ensuring each user has both train and test data
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        
        # Group by user
        grouped = filtered_df.groupby('user_idx')
        
        for user_idx, user_ratings in grouped:
            # Sort by date to ensure chronological order
            user_ratings = user_ratings.sort_values('date')
            
            # Split this user's ratings into train/test
            # Keep at least 3 ratings in training
            train_size = max(3, int(len(user_ratings) * (1 - test_size)))
            
            user_train = user_ratings.iloc[:train_size]
            user_test = user_ratings.iloc[train_size:]
            
            # Ensure test set has at least one item
            if len(user_test) == 0:
                # If not enough ratings, put one in test
                user_test = user_ratings.iloc[-1:]
                user_train = user_ratings.iloc[:-1]
            
            train_df = pd.concat([train_df, user_train])
            test_df = pd.concat([test_df, user_test])
        
        print(f"Training set: {len(train_df)}, Testing set: {len(test_df)}")
        print(f"Number of users in train: {train_df['user_idx'].nunique()}, in test: {test_df['user_idx'].nunique()}")
        
        # Verify that each user in test set has data in train set
        test_users = set(test_df['user_idx'].unique())
        train_users = set(train_df['user_idx'].unique())
        users_missing_from_train = test_users - train_users
        
        if users_missing_from_train:
            print(f"WARNING: {len(users_missing_from_train)} users in test set have no data in train set!")
        
        return train_df, test_df, user_to_idx, business_to_idx
    
    def save_user_cf_model(self, user_item_matrix, user_similarity):
        """Save User-based CF model components"""
        print("Saving User-based CF model...")
        model_path = os.path.join(self.models_folder, 'user_cf_model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        # Save user-item matrix
        user_item_matrix.to_pickle(os.path.join(model_path, 'user_item_matrix.pkl'))
        
        # Save user similarity matrix
        joblib.dump(user_similarity, os.path.join(model_path, 'user_similarity.pkl'))
        
        print(f"User-based CF model saved to {model_path}")
    
    def save_item_cf_model(self, user_item_matrix, item_similarity):
        """Save Item-based CF model components"""
        print("Saving Item-based CF model...")
        model_path = os.path.join(self.models_folder, 'item_cf_model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        # Save user-item matrix
        user_item_matrix.to_pickle(os.path.join(model_path, 'user_item_matrix.pkl'))
        
        # Save item similarity matrix
        joblib.dump(item_similarity, os.path.join(model_path, 'item_similarity.pkl'))
        
        print(f"Item-based CF model saved to {model_path}")
    
    def save_mappings(self, user_to_idx, business_to_idx):
        """Save ID to index mappings"""
        print("Saving ID to index mappings...")
        mappings_path = os.path.join(self.models_folder, 'mappings')
        if not os.path.exists(mappings_path):
            os.makedirs(mappings_path)
            
        # Save user mapping
        with open(os.path.join(mappings_path, 'user_to_idx.pkl'), 'wb') as f:
            pickle.dump(user_to_idx, f)
            
        # Save business mapping
        with open(os.path.join(mappings_path, 'business_to_idx.pkl'), 'wb') as f:
            pickle.dump(business_to_idx, f)
            
        # Also create reverse mappings for convenience
        idx_to_user = {idx: user for user, idx in user_to_idx.items()}
        idx_to_business = {idx: business for business, idx in business_to_idx.items()}
        
        with open(os.path.join(mappings_path, 'idx_to_user.pkl'), 'wb') as f:
            pickle.dump(idx_to_user, f)
            
        with open(os.path.join(mappings_path, 'idx_to_business.pkl'), 'wb') as f:
            pickle.dump(idx_to_business, f)
            
        print(f"Mappings saved to {mappings_path}")
    
    def save_hybrid_model_weights(self, alpha=0.7, beta=0.2, gamma=0.1):
        """Save hybrid model weights"""
        print("Saving hybrid model weights...")
        model_path = os.path.join(self.models_folder, 'hybrid_model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        weights = {
            'alpha': alpha,  # Weight for user-based CF
            'beta': beta,    # Weight for item-based CF
            'gamma': gamma   # Weight for content-based/graph-based
        }
        
        with open(os.path.join(model_path, 'weights.pkl'), 'wb') as f:
            pickle.dump(weights, f)
            
        print(f"Hybrid model weights saved to {model_path}")
    
    def save_graph_model_queries(self):
        """Save graph model queries for reproducibility"""
        print("Saving graph model queries...")
        model_path = os.path.join(self.models_folder, 'graph_model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        # Define the queries used for graph-based recommendations
        queries = {
            'category_based': """
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
            """,
            
            'collaborative': """
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
            """
        }
        
        with open(os.path.join(model_path, 'queries.pkl'), 'wb') as f:
            pickle.dump(queries, f)
            
        print(f"Graph model queries saved to {model_path}")
    
    def save_evaluation_metrics(self, metrics, metrics_table):
        """Save evaluation metrics"""
        print("Saving evaluation metrics...")
        eval_path = os.path.join(self.models_folder, 'evaluation')
        if not os.path.exists(eval_path):
            os.makedirs(eval_path)
            
        # Save metrics as pickle
        with open(os.path.join(eval_path, 'metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)
            
        # Save metrics table as CSV
        metrics_table.to_csv(os.path.join(eval_path, 'metrics_table.csv'))
        
        print(f"Evaluation metrics saved to {eval_path}")
    
    def load_user_cf_model(self):
        """Load User-based CF model components"""
        model_path = os.path.join(self.models_folder, 'user_cf_model')
        if not os.path.exists(model_path):
            print("User-based CF model not found")
            return None, None
        
        user_item_matrix = pd.read_pickle(os.path.join(model_path, 'user_item_matrix.pkl'))
        user_similarity = joblib.load(os.path.join(model_path, 'user_similarity.pkl'))
        
        print(f"User-based CF model loaded from {model_path}")
        return user_item_matrix, user_similarity
    
    def load_item_cf_model(self):
        """Load Item-based CF model components"""
        model_path = os.path.join(self.models_folder, 'item_cf_model')
        if not os.path.exists(model_path):
            print("Item-based CF model not found")
            return None, None
        
        user_item_matrix = pd.read_pickle(os.path.join(model_path, 'user_item_matrix.pkl'))
        item_similarity = joblib.load(os.path.join(model_path, 'item_similarity.pkl'))
        
        print(f"Item-based CF model loaded from {model_path}")
        return user_item_matrix, item_similarity
    
    def load_mappings(self):
        """Load ID to index mappings"""
        mappings_path = os.path.join(self.models_folder, 'mappings')
        if not os.path.exists(mappings_path):
            print("Mappings not found")
            return None, None, None, None
        
        with open(os.path.join(mappings_path, 'user_to_idx.pkl'), 'rb') as f:
            user_to_idx = pickle.load(f)
            
        with open(os.path.join(mappings_path, 'business_to_idx.pkl'), 'rb') as f:
            business_to_idx = pickle.load(f)
            
        with open(os.path.join(mappings_path, 'idx_to_user.pkl'), 'rb') as f:
            idx_to_user = pickle.load(f)
            
        with open(os.path.join(mappings_path, 'idx_to_business.pkl'), 'rb') as f:
            idx_to_business = pickle.load(f)
            
        print(f"Mappings loaded from {mappings_path}")
        return user_to_idx, business_to_idx, idx_to_user, idx_to_business

# Let's update the user_based_cf method to save the model
    def user_based_cf(self, train_df, test_users, k=10, save_model=True):
        """
        User-based collaborative filtering
        Recommends based on similar users' preferences
        
        Parameters:
        - train_df: Training data containing user_idx, business_idx, rating
        - test_users: User indices to make predictions for
        - k: Number of similar users to consider
        - save_model: Whether to save the model components
        
        Returns:
        - Dictionary of recommendations for each user
        """
        print("Running user-based collaborative filtering...")
        start_time = time.time()
        
        # Create user-item rating matrix
        user_item_matrix = pd.pivot_table(
            train_df, 
            values='rating', 
            index='user_idx', 
            columns='business_idx',
            fill_value=0
        )
        
        # Calculate user similarity (cosine similarity)
        from sklearn.metrics.pairwise import cosine_similarity
        user_similarity = cosine_similarity(user_item_matrix)
        
        # Save model if requested
        if save_model:
            self.save_user_cf_model(user_item_matrix, user_similarity)
        
        # For each test user, find similar users and recommend items
        recommendations = {}
        
        for user_idx in test_users:
            if user_idx >= len(user_similarity):
                continue  # Skip users not in training set
                
            # Get similarity scores for this user
            sim_scores = user_similarity[user_idx]
            
            # Get top-k similar users (excluding self)
            similar_users = np.argsort(sim_scores)[::-1][1:k+1]
            
            # Get items rated by similar users but not by the target user
            user_rated_items = set(train_df[train_df['user_idx'] == user_idx]['business_idx'])
            candidate_items = {}
            
            for sim_user in similar_users:
                # Weight of this similar user (similarity score)
                weight = sim_scores[sim_user]
                
                # Items rated by this similar user
                sim_user_ratings = train_df[train_df['user_idx'] == sim_user]
                
                for _, row in sim_user_ratings.iterrows():
                    item = row['business_idx']
                    rating = row['rating']
                    
                    # Skip items already rated by target user
                    if item in user_rated_items:
                        continue
                        
                    # Weighted rating
                    if item not in candidate_items:
                        candidate_items[item] = {'weighted_sum': 0, 'similarity_sum': 0}
                        
                    candidate_items[item]['weighted_sum'] += weight * rating
                    candidate_items[item]['similarity_sum'] += weight
            
            # Calculate predicted ratings
            user_recommendations = {}
            for item, values in candidate_items.items():
                if values['similarity_sum'] > 0:
                    predicted_rating = values['weighted_sum'] / values['similarity_sum']
                    user_recommendations[item] = predicted_rating
            
            # Sort recommendations by predicted rating
            sorted_recommendations = sorted(
                user_recommendations.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            recommendations[user_idx] = sorted_recommendations
        
        end_time = time.time()
        print(f"User-based CF completed in {end_time - start_time:.2f} seconds")
        return recommendations
        

    # Let's update the item_based_cf method to save the model
    def item_based_cf(self, train_df, test_users, k=10, save_model=True):
        """
        Item-based collaborative filtering
        Recommends items similar to ones the user liked
        
        Parameters:
        - train_df: Training data containing user_idx, business_idx, rating
        - test_users: User indices to make predictions for
        - k: Number of similar items to consider
        - save_model: Whether to save the model components
        
        Returns:
        - Dictionary of recommendations for each user
        """
        print("Running item-based collaborative filtering...")
        start_time = time.time()
        
        # Create user-item rating matrix
        user_item_matrix = pd.pivot_table(
            train_df, 
            values='rating', 
            index='user_idx', 
            columns='business_idx',
            fill_value=0
        )
        
        # Calculate item similarity (cosine similarity)
        from sklearn.metrics.pairwise import cosine_similarity
        item_similarity = cosine_similarity(user_item_matrix.T)  # Transpose for item-item similarity
        
        # Save model if requested
        if save_model:
            self.save_item_cf_model(user_item_matrix, item_similarity)
        
        # For each test user, recommend items
        recommendations = {}

        for user_idx in test_users:
            # Get items rated by this user
            user_rated_items = train_df[train_df['user_idx'] == user_idx]
            
            if len(user_rated_items) == 0:
                continue  # Skip users with no ratings
            
            # Calculate predicted ratings for unrated items
            candidate_items = {}
            
            for _, row in user_rated_items.iterrows():
                item = row['business_idx']
                rating = row['rating']
                
                # Get similar items
                if item >= len(item_similarity):
                    continue  # Skip items not in training set
                    
                sim_scores = item_similarity[item]
                
                # Get items not rated by the user
                all_items = set(range(item_similarity.shape[0]))
                user_rated_set = set(user_rated_items['business_idx'])
                unrated_items = all_items - user_rated_set
                
                for unrated_item in unrated_items:
                    if unrated_item >= len(sim_scores):
                        continue
                        
                    similarity = sim_scores[unrated_item]
                    
                    if unrated_item not in candidate_items:
                        candidate_items[unrated_item] = {'weighted_sum': 0, 'similarity_sum': 0}
                        
                    candidate_items[unrated_item]['weighted_sum'] += similarity * rating
                    candidate_items[unrated_item]['similarity_sum'] += similarity
            
            # Calculate predicted ratings
            user_recommendations = {}
            for item, values in candidate_items.items():
                if values['similarity_sum'] > 0:
                    predicted_rating = values['weighted_sum'] / values['similarity_sum']
                    user_recommendations[item] = predicted_rating
            
            # Sort recommendations by predicted rating
            sorted_recommendations = sorted(
                user_recommendations.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            recommendations[user_idx] = sorted_recommendations
        
        end_time = time.time()
        print(f"Item-based CF completed in {end_time - start_time:.2f} seconds")
        return recommendations
        

    def graph_based_recommendation(self, test_users, max_items=10, save_model=True):
        """
        Graph-based recommendation using Neo4j's graph algorithms
        
        Parameters:
        - test_users: List of user IDs to generate recommendations for
        - max_items: Maximum number of items to recommend per user
        - save_model: Whether to save the model queries
        
        Returns:
        - Dictionary of recommendations for each user
        """
        print("Running graph-based recommendation...")
        start_time = time.time()
        
        # Save model if requested
        if save_model:
            self.save_graph_model_queries()
        
        recommendations = {}
            
        with self.driver.session() as session:
            for user_id in test_users:
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
                
                user_recs = [dict(record) for record in result]
                
                # If not enough category-based recommendations, supplement with collaborative approach
                if len(user_recs) < max_items:
                    # FIX: Store the user in a variable first and then reuse it in the NOT EXISTS clause
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
                    """, user_id=user_id, remaining=max_items-len(user_recs))
                    
                    collab_recs = [dict(record) for record in collab_result]
                    user_recs.extend(collab_recs)
                
                recommendations[user_id] = user_recs
        
        end_time = time.time()
        print(f"Graph-based recommendation completed in {end_time - start_time:.2f} seconds")
        return recommendations
        
    # Let's update the hybrid_recommendation method to save the weights
    def hybrid_recommendation(self, train_df, test_users, business_features_df, weights=None):
        """
        Hybrid recommendation method combining CF and content-based approaches
        
        Parameters:
        - train_df: Training data
        - test_users: List of user indices to generate recommendations for
        - business_features_df: Business features for content-based filtering
        - weights: Dictionary of weights for each method (default: equal weights)
        
        Returns:
        - Dictionary of user_idx -> [(item_idx, score), ...]
        """
        print("Running user-based collaborative filtering...")
        start_time = time.time()
        user_cf_recs = self.user_based_cf(train_df, test_users)
        print(f"User-based CF completed in {time.time() - start_time:.2f} seconds")
        
        print("Running item-based collaborative filtering...")
        start_time = time.time()
        item_cf_recs = self.item_based_cf(train_df, test_users)
        print(f"Item-based CF completed in {time.time() - start_time:.2f} seconds")
        
        print("Running graph-based recommendation...")
        start_time = time.time()
        
        # For graph-based, need to map indices back to IDs
        user_idx_to_id = {}
        for _, row in train_df.iterrows():
            user_idx_to_id[row['user_idx']] = row['user_id']
        
        # Map test users to their IDs
        test_user_ids = [user_idx_to_id.get(idx) for idx in test_users if idx in user_idx_to_id]
        test_user_ids = [user_id for user_id in test_user_ids if user_id is not None]
        
        # If there are no valid user IDs, set empty graph recommendations
        if not test_user_ids:
            print("WARNING: No valid user IDs found for graph-based recommendations")
            graph_recs = {}
        else:
            graph_recs_by_id = self.graph_based_recommendation(test_user_ids)
            
            # Convert graph recommendations back to indices format
            id_to_idx = {v: k for k, v in user_idx_to_id.items() if v is not None}
            
            # Create a mapping from business_id to business_idx
            business_id_to_idx = {}
            for _, row in train_df.iterrows():
                business_id_to_idx[row['business_id']] = row['business_idx']
            
            graph_recs = {}
            for user_id, recs in graph_recs_by_id.items():
                if user_id not in id_to_idx:
                    continue
                    
                user_idx = id_to_idx[user_id]
                graph_recs[user_idx] = []
                
                for rec in recs:
                    business_id = rec['business_id']
                    if business_id in business_id_to_idx:
                        business_idx = business_id_to_idx[business_id]
                        # Normalize score to be between 1-5
                        score = min(5, max(1, rec['avg_rating']))
                        graph_recs[user_idx].append((business_idx, score))
        
        print(f"Graph-based recommendation completed in {time.time() - start_time:.2f} seconds")
        
        # Default weights
        if weights is None:
            weights = {'user_cf': 0.3, 'item_cf': 0.3, 'graph': 0.4}
        
        # Save hybrid model weights
        print("Saving hybrid model weights...")
        self.save_hybrid_model_weights(weights)
        
        # Combine recommendations
        hybrid_recs = {}
        
        for user in test_users:
            # Get recommendations from each method
            user_cf = user_cf_recs.get(user, [])
            item_cf = item_cf_recs.get(user, [])
            graph = graph_recs.get(user, [])
            
            # Skip if we have no recommendations
            if not user_cf and not item_cf and not graph:
                continue
            
            # Create a dictionary to combine scores
            combined_scores = {}
            
            # Add weighted user-based CF scores
            for item, score in user_cf:
                if item not in combined_scores:
                    combined_scores[item] = 0
                combined_scores[item] += weights['user_cf'] * score
            
            # Add weighted item-based CF scores
            for item, score in item_cf:
                if item not in combined_scores:
                    combined_scores[item] = 0
                combined_scores[item] += weights['item_cf'] * score
            
            # Add weighted graph-based scores
            for item, score in graph:
                if item not in combined_scores:
                    combined_scores[item] = 0
                combined_scores[item] += weights['graph'] * score
            
            # Convert back to list of (item, score) tuples and sort by score
            user_recs = [(item, score) for item, score in combined_scores.items()]
            user_recs.sort(key=lambda x: x[1], reverse=True)
            
            hybrid_recs[user] = user_recs
        
        return hybrid_recs
    
    def popularity_based_recommendation(self, train_df, test_users, top_n=10):
        """
        Simple popularity-based recommendation as a baseline
        
        Parameters:
        - train_df: Training data
        - test_users: List of user indices to generate recommendations for
        - top_n: Number of top items to recommend
        
        Returns:
        - Dictionary of user_idx -> [(item_idx, score), ...]
        """
        # Group by business and calculate average rating and count
        business_stats = train_df.groupby('business_idx').agg(
            avg_rating=('rating', 'mean'),
            count=('rating', 'count')
        ).reset_index()
        
        # Sort by count (popularity) and then by average rating
        business_stats = business_stats.sort_values(['count', 'avg_rating'], ascending=[False, False])
        
        # Get top-N most popular items
        popular_items = business_stats.head(top_n)
        
        # For each user, recommend the same popular items
        recommendations = {}
        for user in test_users:
            # Get items this user hasn't rated in training set
            user_rated_items = set(train_df[train_df['user_idx'] == user]['business_idx'])
            
            # Filter popular items not rated by this user
            user_recs = []
            for _, item in popular_items.iterrows():
                if item['business_idx'] not in user_rated_items:
                    user_recs.append((item['business_idx'], item['avg_rating']))
                    
            recommendations[user] = user_recs
            
        return recommendations
    
    def debug_recommendations(self, user_cf_recs, item_cf_recs, graph_recs, popularity_recs, hybrid_recs, test_df, n_users=3):
        """
        Debug function to print sample recommendations vs actual test items
        
        Parameters:
        - *_recs: Various recommendation dictionaries
        - test_df: Test data
        - n_users: Number of users to debug
        """
        print("\nDEBUG: Sample Recommendations vs Test Items")
        print("=" * 80)
        
        # Get users that have recommendations from most methods
        common_users = set(user_cf_recs.keys()) & set(item_cf_recs.keys()) & set(graph_recs.keys()) & set(popularity_recs.keys())
        if len(common_users) < n_users:
            common_users = set(popularity_recs.keys())  # Fallback to users with popularity recs
        
        debug_users = list(common_users)[:n_users]
        
        for user in debug_users:
            print(f"\nUser {user}:")
            
            # Get user's test items
            test_items = test_df[test_df['user_idx'] == user]
            print(f"  Actual test items ({len(test_items)}):")
            for _, item in test_items.iterrows():
                print(f"    Business {item['business_idx']}: Rating {item['rating']}")
            
            # Print recommendations from each method
            for method_name, recs in [
                ("User-CF", user_cf_recs.get(user, [])),
                ("Item-CF", item_cf_recs.get(user, [])),
                ("Graph", graph_recs.get(user, [])),
                ("Popularity", popularity_recs.get(user, [])),
                ("Hybrid", hybrid_recs.get(user, []))
            ]:
                print(f"  {method_name} recommendations (top 5):")
                for business, score in recs[:5]:
                    # Check if it's in test set
                    match = test_items[test_items['business_idx'] == business]
                    if not match.empty:
                        actual_rating = match.iloc[0]['rating']
                        print(f"    Business {business}: Score {score:.2f}, Actual Rating: {actual_rating} (HIT)")
                    else:
                        print(f"    Business {business}: Score {score:.2f}")
                        
        print("=" * 80)
    
    # Evaluation functions
    ## def evaluate_recommendations(self, recommendations, test_df, top_n=10):
    def evaluate_recommendations(self, recommendations, test_df, top_n=10):
        """
        Improved evaluation of recommendations against test data
        
        Parameters:
        - recommendations: Dictionary of user_idx -> [(item_idx, score), ...]
        - test_df: Test dataset
        - top_n: Number of top recommendations to consider
        
        Returns:
        - Dictionary of evaluation metrics
        """
        # Prepare actual ratings from test set
        test_ratings = {}
        for _, row in test_df.iterrows():
            user = row['user_idx']
            item = row['business_idx']
            rating = row['rating']
            
            if user not in test_ratings:
                test_ratings[user] = {}
            test_ratings[user][item] = rating
        
        # Use a lower threshold for what's considered "relevant" (3.0 instead of higher values)
        relevance_threshold = 3.0
        
        # Calculate metrics
        precision_at_n = []
        recall_at_n = []
        ndcg_at_n = []
        mae_values = []
        rmse_values = []
        
        evaluated_users = 0
        users_with_hits = 0
        total_hits = 0
        
        for user, user_recs in recommendations.items():
            if user not in test_ratings or not user_recs:
                continue
            
            evaluated_users += 1
            
            # Get top-N recommendations
            top_recs = user_recs[:top_n]
            rec_items = [item for item, _ in top_recs]
            
            # Get relevant items (rated at or above threshold in test set)
            relevant_items = {item for item, rating in test_ratings[user].items() if rating >= relevance_threshold}
            
            # Calculate precision and recall
            if len(rec_items) > 0:
                hits = len(set(rec_items) & relevant_items)
                total_hits += hits
                
                if hits > 0:
                    users_with_hits += 1
                    
                precision = hits / len(rec_items)
                precision_at_n.append(precision)
                
                if len(relevant_items) > 0:
                    recall = hits / len(relevant_items)
                    recall_at_n.append(recall)
            
            # Calculate NDCG
            if len(relevant_items) > 0 and len(rec_items) > 0:
                # Create graded relevance vector for recommendations
                relevance = np.zeros(len(rec_items))
                for i, item in enumerate(rec_items):
                    if item in test_ratings[user]:
                        # Use the actual rating as relevance score
                        relevance[i] = max(0, test_ratings[user][item] - 2)  # Scale ratings 1-5 to relevance 0-3
                
                # If we have relevant items, calculate NDCG
                if np.sum(relevance) > 0:
                    ndcg = self._calculate_ndcg(relevance)
                    ndcg_at_n.append(ndcg)
            
            # Calculate MAE and RMSE for items in both test set and recommendations
            pred_ratings = []
            true_ratings = []
            
            for item, score in top_recs:
                if item in test_ratings[user]:
                    # Skip inf or NaN scores
                    if np.isnan(score) or np.isinf(score):
                        continue
                        
                    # Clamp predicted scores to the rating range [1-5]
                    clamped_score = min(5, max(1, score))
                    pred_ratings.append(clamped_score)
                    true_ratings.append(test_ratings[user][item])
            
            if len(pred_ratings) > 0:
                mae = np.mean(np.abs(np.array(pred_ratings) - np.array(true_ratings)))
                rmse = np.sqrt(np.mean((np.array(pred_ratings) - np.array(true_ratings))**2))
                
                # Check for valid values
                if not (np.isnan(mae) or np.isinf(mae)):
                    mae_values.append(mae)
                if not (np.isnan(rmse) or np.isinf(rmse)):
                    rmse_values.append(rmse)
        
        # Add diagnostic information
        print(f"Evaluated {evaluated_users} users, {users_with_hits} had at least one hit")
        print(f"Total hits: {total_hits}, Average hits per user: {total_hits/evaluated_users if evaluated_users > 0 else 0:.2f}")
        
        # Also print detailed evaluation stats
        print(f"  Precision@{top_n}: {np.mean(precision_at_n) if precision_at_n else 0:.4f}")
        print(f"  Recall@{top_n}: {np.mean(recall_at_n) if recall_at_n else 0:.4f}")
        print(f"  NDCG@{top_n}: {np.mean(ndcg_at_n) if ndcg_at_n else 0:.4f}")
        print(f"  Hit Rate: {users_with_hits / evaluated_users if evaluated_users > 0 else 0:.4f}")
        
        # Aggregate metrics
        metrics = {
            'precision@N': np.mean(precision_at_n) if precision_at_n else 0,
            'recall@N': np.mean(recall_at_n) if recall_at_n else 0,
            'ndcg@N': np.mean(ndcg_at_n) if ndcg_at_n else 0,
            'coverage': len(recommendations) / len(test_ratings) if test_ratings else 0,
            'hit_rate': users_with_hits / evaluated_users if evaluated_users > 0 else 0,
            'mae': np.mean(mae_values) if mae_values else np.nan,
            'rmse': np.mean(rmse_values) if rmse_values else np.nan
        }
        
        # Calculate F1 score
        if metrics['precision@N'] + metrics['recall@N'] > 0:
            metrics['f1@N'] = 2 * (metrics['precision@N'] * metrics['recall@N']) / (metrics['precision@N'] + metrics['recall@N'])
        else:
            metrics['f1@N'] = 0
            
        return metrics

    def _calculate_ndcg(self, relevance):
        """
        Calculate Normalized Discounted Cumulative Gain
        
        Parameters:
        - relevance: Array of relevance scores
        
        Returns:
        - NDCG value
        """
        # Calculate DCG
        dcg = 0
        for i, rel in enumerate(relevance):
            dcg += (2 ** rel - 1) / np.log2(i + 2)  # i+2 because i is 0-indexed
        
        # Calculate ideal DCG
        ideal_relevance = np.sort(relevance)[::-1]  # Sort in descending order
        idcg = 0
        for i, rel in enumerate(ideal_relevance):
            idcg += (2 ** rel - 1) / np.log2(i + 2)
        
        # Return NDCG
        return dcg / idcg if idcg > 0 else 0

    def compare_recommendation_methods(self, train_df, test_df, business_features_df, user_sample=100, top_n=10):
        """
        Compare different recommendation methods with improved evaluation
        
        Parameters:
        - train_df: Training data
        - test_df: Test data
        - business_features_df: Business features
        - user_sample: Number of users to sample for evaluation
        - top_n: Number of top recommendations to evaluate
        
        Returns:
        - Dictionary of evaluation results for each method
        """
        # Only sample users who have ratings in the test set
        test_user_set = set(test_df['user_idx'].unique())
        train_user_set = set(train_df['user_idx'].unique())
        eligible_users = list(test_user_set & train_user_set)
        
        if len(eligible_users) == 0:
            print("ERROR: No users found with ratings in both train and test sets")
            return {}
        
        # Sample users to evaluate (users must have ratings in the test set)
        test_users = np.random.choice(
            eligible_users, 
            min(user_sample, len(eligible_users)), 
            replace=False
        )
        
        print(f"Evaluating recommendations for {len(test_users)} users")
        
        # Verify that our selected users have rating data in the test set
        test_user_ratings = {}
        for _, row in test_df.iterrows():
            user = row['user_idx']
            if user in test_users:
                if user not in test_user_ratings:
                    test_user_ratings[user] = []
                test_user_ratings[user].append(row['business_idx'])
        
        users_with_data = len(test_user_ratings)
        print(f"Found {users_with_data} users with test data out of {len(test_users)} sampled users")
        
        if users_with_data == 0:
            print("ERROR: No sampled users have ratings in the test set")
            return {}
        
        # Generate recommendations using different methods
        print("Running user-based collaborative filtering...")
        start_time = time.time()
        user_cf_recs = self.user_based_cf(train_df, test_users)
        print(f"User-based CF completed in {time.time() - start_time:.2f} seconds")
        
        print("Running item-based collaborative filtering...")
        start_time = time.time()
        item_cf_recs = self.item_based_cf(train_df, test_users)
        print(f"Item-based CF completed in {time.time() - start_time:.2f} seconds")
        
        print("Running graph-based recommendation...")
        start_time = time.time()
        
        # For graph-based, need to map indices back to IDs
        user_idx_to_id = {}
        for _, row in train_df.iterrows():
            user_idx_to_id[row['user_idx']] = row['user_id']
        
        test_user_ids = [user_idx_to_id.get(idx) for idx in test_users if idx in user_idx_to_id]
        
        # Filter out None values
        test_user_ids = [user_id for user_id in test_user_ids if user_id is not None]
        
        # If there are no valid user IDs, set empty graph recommendations
        if not test_user_ids:
            print("WARNING: No valid user IDs found for graph-based recommendations")
            graph_recs = {}
        else:
            graph_recs_by_id = self.graph_based_recommendation(test_user_ids)
            
            # Convert graph recommendations back to indices format for evaluation
            id_to_idx = {v: k for k, v in user_idx_to_id.items() if v is not None}
            
            # Create a mapping from business_id to business_idx
            business_id_to_idx = {}
            for _, row in train_df.iterrows():
                business_id_to_idx[row['business_id']] = row['business_idx']
            
            graph_recs = {}
            for user_id, recs in graph_recs_by_id.items():
                if user_id not in id_to_idx:
                    continue
                    
                user_idx = id_to_idx[user_id]
                graph_recs[user_idx] = []
                
                for rec in recs:
                    business_id = rec['business_id']
                    if business_id in business_id_to_idx:
                        business_idx = business_id_to_idx[business_id]
                        # Normalize score to be between 1-5
                        score = min(5, max(1, rec['avg_rating']))
                        graph_recs[user_idx].append((business_idx, score))
        
        print(f"Graph-based recommendation completed in {time.time() - start_time:.2f} seconds")
        
        # Add a popularity-based recommendation as baseline
        print("Running popularity-based recommendation (baseline)...")
        popularity_recs = self.popularity_based_recommendation(train_df, test_users, top_n)
        
        # Check if we have valid recommendations to evaluate
        for method_name, recs in [("User-based CF", user_cf_recs), 
                                ("Item-based CF", item_cf_recs), 
                                ("Graph-based", graph_recs),
                                ("Popularity", popularity_recs)]:
            total_recs = sum(len(user_recs) for user_recs in recs.values())
            print(f"{method_name}: {len(recs)} users with recommendations, {total_recs} total recommendations")
        
        print("Running hybrid recommendation...")
        start_time = time.time()
        hybrid_recs = self.hybrid_recommendation(train_df, test_users, business_features_df)
        print(f"Hybrid recommendation completed in {time.time() - start_time:.2f} seconds")
        
        # Evaluate each method
        metrics = {
            'User-based CF': self.evaluate_recommendations(user_cf_recs, test_df, top_n),
            'Item-based CF': self.evaluate_recommendations(item_cf_recs, test_df, top_n),
            'Graph-based': self.evaluate_recommendations(graph_recs, test_df, top_n),
            'Popularity': self.evaluate_recommendations(popularity_recs, test_df, top_n),
            'Hybrid': self.evaluate_recommendations(hybrid_recs, test_df, top_n)
        }
        
        # Debug: Print sample recommendations vs actual test items for a few users
        self.debug_recommendations(user_cf_recs, item_cf_recs, graph_recs, popularity_recs, hybrid_recs, test_df, n_users=3)
        
        return metrics

    def visualize_metrics(self, metrics):
        """
        Visualize evaluation metrics
        
        Parameters:
        - metrics: Dictionary of evaluation results from compare_recommendation_methods
        """
        # Create dataframe from metrics
        methods = list(metrics.keys())
        
        # Metrics to plot
        metric_names = ['precision@N', 'recall@N', 'f1@N', 'ndcg@N', 'coverage']
        
        # Create data for plotting
        plot_data = {}
        for metric_name in metric_names:
            plot_data[metric_name] = [metrics[method][metric_name] for method in methods]
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot each metric
        for i, metric_name in enumerate(metric_names):
            axes[i].bar(methods, plot_data[metric_name])
            axes[i].set_title(f'{metric_name}')
            # Get the maximum value, ensuring it's a valid number
            max_val = max([v for v in plot_data[metric_name] if not (np.isnan(v) or np.isinf(v))], default=0.1)
            axes[i].set_ylim(0, max_val * 1.2 + 0.01)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add values on top of bars
            for j, value in enumerate(plot_data[metric_name]):
                if not (np.isnan(value) or np.isinf(value)):
                    axes[i].text(j, value + 0.005, f'{value:.3f}', ha='center')
        
        # Plot error metrics (MAE and RMSE)
        error_metrics = ['mae', 'rmse']
        error_data = {}
        for metric_name in error_metrics:
            error_data[metric_name] = [metrics[method][metric_name] for method in methods]
        
        # Filter out NaN and Inf values for RMSE
        valid_rmse = [v for v in error_data['rmse'] if not (np.isnan(v) or np.isinf(v))]
        
        if valid_rmse:  # Check if there are any valid RMSE values
            max_rmse = max(valid_rmse)
            axes[5].bar(methods, [v if not (np.isnan(v) or np.isinf(v)) else 0 for v in error_data['rmse']])
            axes[5].set_title('RMSE (lower is better)')
            axes[5].set_ylim(0, max_rmse * 1.2 + 0.01)
            axes[5].tick_params(axis='x', rotation=45)
            
            for j, value in enumerate(error_data['rmse']):
                if not (np.isnan(value) or np.isinf(value)):
                    axes[5].text(j, value + 0.005, f'{value:.3f}', ha='center')
        else:
            axes[5].set_title('RMSE (No valid data)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_folder, 'recommendation_metrics.png'))
        plt.close()
        
        # More detailed table of metrics
        metrics_table = pd.DataFrame(metrics).T
        
        # Replace NaN and Inf values with a placeholder for display
        metrics_table = metrics_table.replace([np.inf, -np.inf], np.nan)
        metrics_table = metrics_table.fillna('N/A')
        
        # Round only numeric values
        for col in metrics_table.columns:
            metrics_table[col] = metrics_table[col].apply(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
        
        return metrics_table
    
def main():
    # Neo4j connection details
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password" 
    
    # Create recommender with a specified models folder
    models_folder = 'models'
    recommender = YelpRecommender(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, models_folder=models_folder)
    
    try:
        # Fetch data from Neo4j
        print("Fetching data from Neo4j...")
        ratings_df = recommender.fetch_user_business_ratings()
        business_features_df = recommender.fetch_business_features()
        
        # Print some stats about the data
        print(f"Total ratings: {len(ratings_df)}")
        print(f"Unique users: {ratings_df['user_id'].nunique()}")
        print(f"Unique businesses: {ratings_df['business_id'].nunique()}")
        print(f"Rating distribution: {ratings_df['rating'].value_counts().sort_index()}")
        
        # Preprocess data (with improved preprocessing)
        train_df, test_df, user_to_idx, business_to_idx = recommender.preprocess_data(ratings_df)
        
        # Save the mappings
        recommender.save_mappings(user_to_idx, business_to_idx)
        
        # Compare recommendation methods (with a smaller user sample to ensure good coverage)
        print("Comparing recommendation methods...")
        metrics = recommender.compare_recommendation_methods(
            train_df, test_df, business_features_df, user_sample=50
        )
        
        # Visualize results
        metrics_table = recommender.visualize_metrics(metrics)
        print("\nRecommendation System Evaluation Results:")
        print(metrics_table)
        
        # Save evaluation metrics
        recommender.save_evaluation_metrics(metrics, metrics_table)
        
        # Save results to CSV in the models folder
        metrics_table.to_csv(os.path.join(recommender.models_folder, 'recommendation_metrics.csv'))
        
        print(f"\nRecommendation system evaluation complete. Results and models saved to {recommender.models_folder}")
        
        # [Rest of the main function stays the same]
        
        print("\nGenerating sample recommendations for a random user...")
            
        # Get a random user from the test set
        sample_user_idx = np.random.choice(test_df['user_idx'].unique())
        sample_user_id = None
    
        # Find the user_id for this user_idx
        for user_id, idx in user_to_idx.items():
            if idx == sample_user_idx:
                sample_user_id = user_id
                break
        
        if sample_user_id:
            # Get hybrid recommendations
            user_recommendations = recommender.graph_based_recommendation([sample_user_id], max_items=10)
            
            print(f"\nTop 10 recommendations for user {sample_user_id}:")
            if sample_user_id in user_recommendations:
                for i, rec in enumerate(user_recommendations[sample_user_id][:10], 1):
                    print(f"{i}. {rec['name']} (Rating: {rec['avg_rating']}, Score: {rec['score']})")
                    print(f"   Categories: {', '.join(rec['matched_categories'][:3])}")
            else:
                print("No recommendations found for this user.")

    finally:
        recommender.close()

if __name__ == "__main__":
    main()