import pandas as pd
import numpy as np
import pickle
import os
import json
from neo4j import GraphDatabase
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class YelpRecommendationSystem:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # Store model parameters
        self.parameters = {
            'similarity_threshold': 0.3,
            'cf_weight': 0.6,
            'cb_weight': 0.4
        }
        # Cache for storing user preferences and similarity data
        self.user_preferences = {}
        self.user_similarities = {}
        
    def close(self):
        self.driver.close()
        
    def fetch_user_business_ratings(self):
        """
        Fetch all user-business ratings from the database
        Returns DataFrame with user_id, business_id, stars columns
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User)-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
                RETURN u.user_id AS user_id, b.business_id AS business_id, r.stars AS stars
                LIMIT 100000  // Added limit to prevent memory issues
            """)
            
            ratings = []
            for record in result:
                ratings.append({
                    'user_id': record['user_id'],
                    'business_id': record['business_id'],
                    'stars': record['stars']
                })
            
            return pd.DataFrame(ratings)
    
    def build_user_preference_model(self, save_path='models', batch_size=1000):
        """
        Build and save user preference models for faster recommendations
        Using batched processing to avoid memory issues
        """
        # Ensure the directory exists
        os.makedirs(save_path, exist_ok=True)
        
        with self.driver.session() as session:
            # Build user-category preferences
            print("Building user-category preferences...")
            user_category_result = session.run("""
                MATCH (u:User)-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)-[:IN_CATEGORY]->(c:Category)
                RETURN u.user_id AS user_id, c.name AS category, 
                       count(b) AS frequency, avg(r.stars) AS avg_rating
            """)
            
            user_category_prefs = defaultdict(dict)
            for record in user_category_result:
                user_id = record['user_id']
                category = record['category']
                weight = record['frequency'] * record['avg_rating']
                user_category_prefs[user_id][category] = weight
            
            # Save the user preferences
            self.user_preferences = dict(user_category_prefs)
            with open(os.path.join(save_path, 'user_category_preferences.pkl'), 'wb') as f:
                pickle.dump(self.user_preferences, f)
                
            # Build user-user similarities in batches
            print("Building user similarity model in batches...")
            
            # First, get a list of all users with sufficient reviews
            active_users = session.run("""
                MATCH (u:User)-[:WROTE]->(r:Review)
                WITH u, count(r) AS review_count
                WHERE review_count >= 5
                RETURN u.user_id AS user_id
                ORDER BY review_count DESC
                LIMIT 5000  // Limit to most active users
            """).values()
            
            active_users = [item[0] for item in active_users]
            print(f"Processing similarities for {len(active_users)} active users")
            
            user_similarities = defaultdict(dict)
            
            # Process in batches to avoid memory issues
            for i in range(0, len(active_users), batch_size):
                batch_users = active_users[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}/{len(active_users)//batch_size + 1} ({len(batch_users)} users)")
                
                # Use a more efficient query with SKIP and LIMIT
                for user_idx, user_id in enumerate(batch_users):
                    if user_idx % 100 == 0:
                        print(f"  Processing user {user_idx}/{len(batch_users)}")
                    
                    # Get users with similar ratings more efficiently
                    user_similarity_result = session.run("""
                        MATCH (u1:User {user_id: $user_id})-[:WROTE]->(r1:Review)-[:ABOUT]->(b:Business)
                        WITH u1, b, r1.stars AS r1_stars
                        MATCH (b)<-[:ABOUT]-(r2:Review)<-[:WROTE]-(u2:User)
                        WHERE u1 <> u2
                        WITH u2.user_id AS user2, COUNT(b) AS common_businesses, 
                             SUM(ABS(r1_stars - r2.stars)) AS total_diff
                        WHERE common_businesses >= 3  // Reduced threshold
                        WITH user2, common_businesses, total_diff / common_businesses AS avg_diff
                        ORDER BY common_businesses DESC, avg_diff ASC
                        LIMIT 20  // Only keep top 20 similar users
                        RETURN user2, common_businesses, avg_diff
                    """, user_id=user_id)
                    
                    # Process results for this user
                    for record in user_similarity_result:
                        user2 = record['user2']
                        common = record['common_businesses']
                        avg_diff = record['avg_diff']
                        
                        # Calculate similarity score (higher is better)
                        similarity = (1 / (1 + avg_diff)) * np.log1p(common)
                        
                        # Only store if similarity is above threshold
                        if similarity > 0.2:  # Threshold to keep model size manageable
                            user_similarities[user_id][user2] = similarity
            
            # Save the user similarities
            self.user_similarities = dict(user_similarities)
            with open(os.path.join(save_path, 'user_similarities.pkl'), 'wb') as f:
                pickle.dump(self.user_similarities, f)
                
            # Save model parameters
            with open(os.path.join(save_path, 'model_parameters.json'), 'w') as f:
                json.dump(self.parameters, f)
                
            print(f"Models saved to {save_path}")
            
    def load_models(self, load_path='models'):
        """
        Load saved models
        """
        try:
            # Load user category preferences
            with open(os.path.join(load_path, 'user_category_preferences.pkl'), 'rb') as f:
                self.user_preferences = pickle.load(f)
                
            # Load user similarities
            with open(os.path.join(load_path, 'user_similarities.pkl'), 'rb') as f:
                self.user_similarities = pickle.load(f)
                
            # Load parameters
            with open(os.path.join(load_path, 'model_parameters.json'), 'r') as f:
                self.parameters = json.load(f)
                
            print(f"Models loaded from {load_path}")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def collaborative_filtering(self, target_user_id, n=10, similarity_threshold=None):
        """
        User-based collaborative filtering
        Returns top n recommended businesses for the target user
        """
        if similarity_threshold is None:
            similarity_threshold = self.parameters['similarity_threshold']
            
        with self.driver.session() as session:
            # First, get all the businesses the target user has already rated
            rated_businesses = session.run("""
                MATCH (u:User {user_id: $user_id})-[:WROTE]->()-[:ABOUT]->(b:Business)
                RETURN b.business_id AS business_id
            """, user_id=target_user_id).values()
            
            rated_businesses = [item[0] for item in rated_businesses]
            
            # Use pre-computed similarities if available
            similar_users = []
            if target_user_id in self.user_similarities:
                for other_user, similarity in sorted(self.user_similarities[target_user_id].items(), 
                                                   key=lambda x: x[1], reverse=True)[:50]:
                    similar_users.append({
                        'other_user_id': other_user,
                        'similarity': similarity
                    })
            
            # If no pre-computed similarities, fetch from database with optimized query
            if not similar_users:
                similar_users_result = session.run("""
                    MATCH (target:User {user_id: $user_id})-[:WROTE]->(r1:Review)-[:ABOUT]->(b:Business)
                    WITH target, b, r1.stars AS r1_stars
                    MATCH (b)<-[:ABOUT]-(r2:Review)<-[:WROTE]-(other:User)
                    WHERE target <> other
                    WITH other.user_id AS other_user_id, 
                         count(b) AS common_businesses,
                         sum(abs(r1_stars - r2.stars)) AS rating_diff
                    WHERE common_businesses >= 3
                    WITH other_user_id, common_businesses, rating_diff / common_businesses AS avg_diff
                    ORDER BY avg_diff ASC, common_businesses DESC
                    LIMIT 50
                    RETURN other_user_id, common_businesses, avg_diff
                """, user_id=target_user_id)
                
                for record in similar_users_result:
                    similar_user_id = record['other_user_id']
                    common_businesses = record['common_businesses']
                    rating_diff = record['avg_diff']
                    
                    # Skip users with too different ratings
                    if rating_diff > similarity_threshold:
                        continue
                    
                    # Calculate similarity score (higher is better)
                    similarity = (1 / (1 + rating_diff)) * common_businesses
                    
                    similar_users.append({
                        'other_user_id': similar_user_id,
                        'similarity': similarity
                    })
            
            # Get recommendations from similar users
            recommendations = []
            
            # Process similar users in smaller batches to avoid memory issues
            batch_size = 10
            for i in range(0, len(similar_users), batch_size):
                batch = similar_users[i:i+batch_size]
                
                for user_data in batch:
                    similar_user_id = user_data['other_user_id']
                    similarity = user_data['similarity']
                    
                    # Get highly-rated businesses from this similar user that target user hasn't rated
                    similar_user_businesses = session.run("""
                        MATCH (u:User {user_id: $similar_user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
                        WHERE r.stars >= 4 AND NOT b.business_id IN $rated_businesses
                        RETURN b.business_id AS business_id, b.name AS name, b.stars AS avg_stars, 
                               r.stars AS user_stars, b.review_count AS review_count
                        LIMIT 20  // Added limit to prevent memory issues
                    """, similar_user_id=similar_user_id, rated_businesses=rated_businesses)
                    
                    for business in similar_user_businesses:
                        # Calculate recommendation score
                        rec_score = similarity * business['user_stars']
                        
                        recommendations.append({
                            'business_id': business['business_id'],
                            'name': business['name'],
                            'avg_stars': business['avg_stars'],
                            'rec_score': rec_score,
                            'review_count': business['review_count']
                        })
            
            # Sort and remove duplicates
            recommendations_df = pd.DataFrame(recommendations)
            if recommendations_df.empty:
                return recommendations_df
                
            recommendations_df = recommendations_df.sort_values('rec_score', ascending=False)
            recommendations_df = recommendations_df.drop_duplicates(subset=['business_id'])
            
            return recommendations_df.head(n)
    
    def content_based_recommendation(self, target_user_id, n=10):
        """
        Content-based filtering based on categories the user has rated highly
        """
        with self.driver.session() as session:
            # Get all businesses already rated by the user
            rated_businesses = session.run("""
                MATCH (u:User {user_id: $user_id})-[:WROTE]->()-[:ABOUT]->(b:Business)
                RETURN b.business_id AS business_id
                LIMIT 1000  // Added limit to prevent memory issues
            """, user_id=target_user_id).values()
            
            rated_businesses = [item[0] for item in rated_businesses]
            
            # Use pre-computed category preferences if available
            category_weights = {}
            if target_user_id in self.user_preferences:
                category_weights = self.user_preferences[target_user_id]
            
            # If no pre-computed preferences, fetch from database with optimized query
            if not category_weights:
                # Find categories the user likes based on high ratings
                liked_categories = session.run("""
                    MATCH (u:User {user_id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)-[:IN_CATEGORY]->(c:Category)
                    WHERE r.stars >= 4
                    RETURN c.name AS category, count(b) AS frequency, avg(r.stars) AS avg_rating
                    ORDER BY frequency DESC, avg_rating DESC
                    LIMIT 10
                """, user_id=target_user_id)
                
                # Store categories and their weights
                for record in liked_categories:
                    category_weights[record['category']] = record['frequency'] * record['avg_rating']
            
            if not category_weights:
                return pd.DataFrame()  # Return empty DataFrame if no preferences found
            
            # Find businesses in those categories that the user hasn't rated
            recommendations = []
            
            # Process categories in batches to avoid memory issues
            category_list = list(category_weights.items())
            batch_size = 5  # Process 5 categories at a time
            
            for i in range(0, len(category_list), batch_size):
                batch_categories = category_list[i:i+batch_size]
                
                for category, weight in batch_categories:
                    category_businesses = session.run("""
                        MATCH (c:Category {name: $category})<-[:IN_CATEGORY]-(b:Business)
                        WHERE NOT b.business_id IN $rated_businesses
                        AND b.stars >= 3.5 AND b.review_count >= 10
                        RETURN b.business_id AS business_id, b.name AS name, 
                               b.stars AS avg_stars, b.review_count AS review_count
                        LIMIT 25
                    """, category=category, rated_businesses=rated_businesses)
                    
                    for business in category_businesses:
                        # Calculate recommendation score
                        rec_score = weight * business['avg_stars']
                        
                        recommendations.append({
                            'business_id': business['business_id'],
                            'name': business['name'],
                            'avg_stars': business['avg_stars'],
                            'category': category,
                            'rec_score': rec_score,
                            'review_count': business['review_count']
                        })
            
            # Sort and remove duplicates
            recommendations_df = pd.DataFrame(recommendations)
            if recommendations_df.empty:
                return recommendations_df
                
            recommendations_df = recommendations_df.sort_values('rec_score', ascending=False)
            recommendations_df = recommendations_df.drop_duplicates(subset=['business_id'])
            
            return recommendations_df.head(n)
    
    def hybrid_recommendation(self, target_user_id, n=10):
        """
        Hybrid recommendation combining collaborative and content-based filtering
        """
        # Get recommendations from both methods
        cf_recs = self.collaborative_filtering(target_user_id, n=n*2)
        cb_recs = self.content_based_recommendation(target_user_id, n=n*2)
        
        # Weight the scores (can be adjusted)
        cf_weight = self.parameters['cf_weight']
        cb_weight = self.parameters['cb_weight']
        
        if not cf_recs.empty:
            cf_recs['score'] = cf_recs['rec_score'] * cf_weight
        
        if not cb_recs.empty:
            cb_recs['score'] = cb_recs['rec_score'] * cb_weight
        
        # Combine recommendations
        combined_recs = pd.concat([cf_recs, cb_recs], ignore_index=True)
        
        if combined_recs.empty:
            # Fallback to popular businesses if no recommendations
            return self.popular_businesses_recommendation(n)
        
        # Aggregate scores for businesses that appear in both methods
        combined_recs = combined_recs.groupby('business_id').agg({
            'name': 'first',
            'avg_stars': 'first',
            'review_count': 'first',
            'score': 'sum'
        }).reset_index()
        
        # Sort by score and return top n
        return combined_recs.sort_values('score', ascending=False).head(n)
    
    def popular_businesses_recommendation(self, n=10, min_stars=4.0):
        """
        Fallback recommendation method using popular highly-rated businesses
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (b:Business)
                WHERE b.stars >= $min_stars AND b.review_count >= 20
                RETURN b.business_id AS business_id, b.name AS name,
                       b.stars AS avg_stars, b.review_count AS review_count,
                       b.stars * log(b.review_count) AS popularity_score
                ORDER BY popularity_score DESC
                LIMIT $limit
            """, min_stars=min_stars, limit=n)
            
            recommendations = []
            for record in result:
                recommendations.append({
                    'business_id': record['business_id'],
                    'name': record['name'],
                    'avg_stars': record['avg_stars'],
                    'review_count': record['review_count'],
                    'score': record['popularity_score']
                })
            
            return pd.DataFrame(recommendations)
    
    def get_user_demographic_recommendations(self, target_user_id, n=10):
        """
        Generate recommendations based on user demographics (users with similar rating patterns)
        """
        with self.driver.session() as session:
            # Get all businesses already rated by the user
            rated_businesses = session.run("""
                MATCH (u:User {user_id: $user_id})-[:WROTE]->()-[:ABOUT]->(b:Business)
                RETURN b.business_id AS business_id
                LIMIT 1000  // Added limit to prevent memory issues
            """, user_id=target_user_id).values()
            
            rated_businesses = [item[0] for item in rated_businesses]
            
            # Find users with similar overall rating patterns (average stars, review count)
            similar_users = session.run("""
                MATCH (target:User {user_id: $user_id})
                MATCH (other:User)
                WHERE target <> other
                AND abs(other.average_stars - target.average_stars) < 0.5
                AND abs(other.review_count - target.review_count) < target.review_count * 0.3
                RETURN other.user_id AS user_id
                LIMIT 50
            """, user_id=target_user_id)
            
            similar_user_ids = [record['user_id'] for record in similar_users]
            
            if not similar_user_ids:
                return self.popular_businesses_recommendation(n)
            
            # Process similar users in batches
            batch_size = 10
            recommendations = []
            
            for i in range(0, len(similar_user_ids), batch_size):
                batch_users = similar_user_ids[i:i+batch_size]
                
                # Find businesses highly rated by this batch of similar users
                result = session.run("""
                    MATCH (u:User)-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
                    WHERE u.user_id IN $batch_users
                    AND NOT b.business_id IN $rated_businesses
                    AND r.stars >= 4
                    WITH b, avg(r.stars) AS avg_user_rating, count(u) AS user_count
                    WHERE user_count >= 2
                    RETURN b.business_id AS business_id, b.name AS name,
                           b.stars AS avg_stars, b.review_count AS review_count,
                           avg_user_rating, user_count,
                           avg_user_rating * log10(user_count + 1) AS score
                    ORDER BY score DESC
                    LIMIT 20
                """, batch_users=batch_users, rated_businesses=rated_businesses)
                
                for record in result:
                    recommendations.append({
                        'business_id': record['business_id'],
                        'name': record['name'],
                        'avg_stars': record['avg_stars'],
                        'avg_user_rating': record['avg_user_rating'],
                        'user_count': record['user_count'],
                        'review_count': record['review_count'],
                        'score': record['score']
                    })
            
            # Convert to DataFrame
            recommendations_df = pd.DataFrame(recommendations)
            if recommendations_df.empty:
                return self.popular_businesses_recommendation(n)
                
            # Sort by score and remove duplicates
            recommendations_df = recommendations_df.sort_values('score', ascending=False)
            recommendations_df = recommendations_df.drop_duplicates(subset=['business_id'])
            
            return recommendations_df.head(n)
    
    def recommend_for_user(self, user_id, method='hybrid', n=10):
        """
        Generate recommendations for a user using the specified method
        """
        methods = {
            'collaborative': self.collaborative_filtering,
            'content': self.content_based_recommendation,
            'hybrid': self.hybrid_recommendation,
            'popular': self.popular_businesses_recommendation,
            'demographic': self.get_user_demographic_recommendations
        }
        
        if method not in methods:
            raise ValueError(f"Method must be one of {list(methods.keys())}")
        
        # Special case for popular method which doesn't require user_id
        if method == 'popular':
            return self.popular_businesses_recommendation(n)
        
        # Call the appropriate recommendation method
        return methods[method](user_id, n)
    
    def update_parameters(self, parameters, save_path='models'):
        """
        Update the model parameters and save them
        """
        self.parameters.update(parameters)
        
        # Save updated parameters
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'model_parameters.json'), 'w') as f:
            json.dump(self.parameters, f)
            
        print(f"Parameters updated and saved to {save_path}")


class RecommendationEvaluation:
    def __init__(self, recommender):
        self.recommender = recommender
        
    def create_train_test_split(self, test_ratio=0.2, min_ratings=10, max_users=1000):
        """
        Split user ratings into training and testing datasets
        With memory optimization limits
        """
        # Fetch all ratings with limit
        ratings_df = self.recommender.fetch_user_business_ratings()
        
        # Get users with sufficient number of ratings
        user_counts = ratings_df['user_id'].value_counts()
        eligible_users = user_counts[user_counts >= min_ratings].index.tolist()
        
        # Limit number of users to evaluate
        if len(eligible_users) > max_users:
            eligible_users = eligible_users[:max_users]
        
        # Filter ratings to only include eligible users
        eligible_ratings = ratings_df[ratings_df['user_id'].isin(eligible_users)]
        
        # Create user-specific train/test splits
        train_data = []
        test_data = []
        
        for user_id in eligible_users:
            user_ratings = eligible_ratings[eligible_ratings['user_id'] == user_id]
            user_train, user_test = train_test_split(user_ratings, test_size=test_ratio, random_state=42)
            
            train_data.append(user_train)
            test_data.append(user_test)
        
        # Combine all users' train/test data
        train_df = pd.concat(train_data)
        test_df = pd.concat(test_data)
        
        return train_df, test_df, eligible_users
    
    def evaluate_rmse(self, test_users, k=10, max_users=50):
        """
        Evaluate using Root Mean Squared Error
        Lower RMSE is better
        """
        # Limit number of users to evaluate
        if len(test_users) > max_users:
            test_users = test_users[:max_users]
            
        all_errors = []
        
        for user_id in test_users:
            # Get recommendations for this user
            recommendations = self.recommender.hybrid_recommendation(user_id, n=k)
            
            if recommendations.empty:
                continue
                
            # Fetch actual ratings for recommended businesses
            with self.recommender.driver.session() as session:
                actual_ratings = session.run("""
                    MATCH (u:User {user_id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
                    WHERE b.business_id IN $business_ids
                    RETURN b.business_id AS business_id, r.stars AS actual_rating
                """, user_id=user_id, business_ids=recommendations['business_id'].tolist())
                
                # Convert to dictionary for easy lookup
                ratings_dict = {record['business_id']: record['actual_rating'] 
                                for record in actual_ratings}
                
                # For each recommendation with an actual rating, compute error
                for _, rec in recommendations.iterrows():
                    business_id = rec['business_id']
                    if business_id in ratings_dict:
                        # Normalize recommendation score to 1-5 scale
                        predicted_rating = min(5, max(1, rec['score']))
                        actual_rating = ratings_dict[business_id]
                        error = (predicted_rating - actual_rating) ** 2
                        all_errors.append(error)
        
        if not all_errors:
            return float('inf')  # Return infinity if no errors calculated
            
        return np.sqrt(np.mean(all_errors))
    
    def evaluate_precision_recall_at_k(self, test_users, k=10, threshold=4, max_users=50):
        """
        Evaluate using precision and recall at k
        """
        # Limit number of users to evaluate
        if len(test_users) > max_users:
            test_users = test_users[:max_users]
            
        precisions = []
        recalls = []
        
        for user_id in test_users:
            # Get recommendations for this user
            recommendations = self.recommender.hybrid_recommendation(user_id, n=k)
            
            if recommendations.empty:
                continue
                
            # Get relevant items for this user (rated >= threshold)
            with self.recommender.driver.session() as session:
                relevant_businesses = session.run("""
                    MATCH (u:User {user_id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
                    WHERE r.stars >= $threshold
                    RETURN b.business_id AS business_id
                """, user_id=user_id, threshold=threshold).values()
                
                relevant_businesses = set([item[0] for item in relevant_businesses])
                
                # Get the recommended business IDs
                recommended_businesses = set(recommendations['business_id'].tolist())
                
                # Calculate relevant recommended items
                relevant_recommended = recommended_businesses.intersection(relevant_businesses)
                
                # Calculate precision and recall
                precision = len(relevant_recommended) / len(recommended_businesses) if recommended_businesses else 0
                recall = len(relevant_recommended) / len(relevant_businesses) if relevant_businesses else 0
                
                precisions.append(precision)
                recalls.append(recall)
        
        # Calculate average precision and recall
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': f1
        }
    
    def map_at_k(self, test_users, k=10, threshold=4, max_users=50):
        """
        Calculate Mean Average Precision at k
        """
        # Limit number of users to evaluate
        if len(test_users) > max_users:
            test_users = test_users[:max_users]
            
        avg_precisions = []
        
        for user_id in test_users:
            # Get recommendations for this user
            recommendations = self.recommender.hybrid_recommendation(user_id, n=k)
            
            if recommendations.empty:
                continue
                
            # Get actual ratings for all businesses
            with self.recommender.driver.session() as session:
                actual_ratings = session.run("""
                    MATCH (u:User {user_id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
                    RETURN b.business_id AS business_id, r.stars AS stars
                """, user_id=user_id)
                
                # Create dictionary of business_id -> rating
                ratings_dict = {record['business_id']: record['stars'] 
                                for record in actual_ratings}
                
                # Calculate precision at each position
                relevant_count = 0
                precisions = []
                
                for i, (_, rec) in enumerate(recommendations.iterrows(), 1):
                    business_id = rec['business_id']
                    if business_id in ratings_dict and ratings_dict[business_id] >= threshold:
                        relevant_count += 1
                        precisions.append(relevant_count / i)
                
                # Calculate average precision for this user
                avg_precision = np.mean(precisions) if precisions else 0
                avg_precisions.append(avg_precision)
        
        # Calculate MAP
        return np.mean(avg_precisions) if avg_precisions else 0
    
    # Completing the evaluate_recommendations method
    def evaluate_recommendations(self, test_ratio=0.2, k=10, min_ratings=10):
        """
        Run a comprehensive evaluation of the recommendation system
        """
        # Create train/test split with limited number of users
        train_df, test_df, test_users = self.create_train_test_split(test_ratio=test_ratio, min_ratings=min_ratings)
        
        print(f"Evaluation using {len(test_users)} users with {len(test_df)} test ratings")
        
        # Calculate RMSE
        rmse = self.evaluate_rmse(test_users, k=k)
        
        # Calculate precision, recall, F1
        precision_recall = self.evaluate_precision_recall_at_k(test_users, k=k)
        
        # Calculate MAP
        map_score = self.map_at_k(test_users, k=k)
        
        # Return all metrics
        metrics = {
            'rmse': rmse,
            'precision': precision_recall['precision'],
            'recall': precision_recall['recall'],
            'f1': precision_recall['f1'],
            'map': map_score
        }
        
        print("Evaluation Results:")
        print(f"RMSE: {rmse:.4f}")
        print(f"Precision@{k}: {precision_recall['precision']:.4f}")
        print(f"Recall@{k}: {precision_recall['recall']:.4f}")
        print(f"F1@{k}: {precision_recall['f1']:.4f}")
        print(f"MAP@{k}: {map_score:.4f}")
        
        return metrics


# Example usage
def main():
    # Neo4j connection details
    uri = "neo4j://localhost:7687"  # Update with your Neo4j URI
    user = "neo4j"                  # Update with your username
    password = "password"           # Update with your password
    
    # Initialize the recommendation system
    print("Initializing Yelp Recommendation System...")
    recommender = YelpRecommendationSystem(uri, user, password)
    
    # Try to load existing models
    if not recommender.load_models():
        print("Building models...")
        recommender.build_user_preference_model(batch_size=500)
    
    # Example 1: Get recommendations for a specific user
    example_user_id = "Wc5L5WJAkEP98hLI8xb9gQ"  # Replace with an actual user ID
    print(f"\nGenerating recommendations for user {example_user_id}...")
    
    # Get recommendations using different methods
    methods = ['collaborative', 'content', 'hybrid', 'demographic']
    for method in methods:
        print(f"\n{method.capitalize()} Filtering Recommendations:")
        recommendations = recommender.recommend_for_user(example_user_id, method=method, n=5)
        if not recommendations.empty:
            print(recommendations[['name', 'avg_stars', 'score']].to_string(index=False))
        else:
            print("No recommendations found using this method.")
    
    # Example 2: Run evaluation
    print("\nRunning evaluation...")
    evaluator = RecommendationEvaluation(recommender)
    metrics = evaluator.evaluate_recommendations(test_ratio=0.2, k=10, min_ratings=5)
    
    # Example 3: Parameter tuning
    print("\nTuning model parameters...")
    best_f1 = metrics['f1']
    best_params = recommender.parameters.copy()
    
    # Try different parameter combinations
    for cf_weight in [0.5, 0.6, 0.7]:
        for cb_weight in [0.5, 0.4, 0.3]:
            for similarity_threshold in [0.2, 0.3, 0.4]:
                # Update parameters
                new_params = {
                    'cf_weight': cf_weight,
                    'cb_weight': cb_weight,
                    'similarity_threshold': similarity_threshold
                }
                recommender.update_parameters(new_params)
                
                # Evaluate with new parameters
                print(f"Testing parameters: {new_params}")
                metrics = evaluator.evaluate_recommendations(test_ratio=0.2, k=10, min_ratings=5)
                
                # Save if better
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_params = new_params.copy()
                    print(f"Found better parameters: {best_params} with F1: {best_f1:.4f}")
    
    # Use the best parameters
    recommender.update_parameters(best_params)
    print(f"\nFinal best parameters: {best_params} with F1: {best_f1:.4f}")
    
    # Example 4: Generate visualizations of recommendation performance
    print("\nGenerating visualizations...")
    
    # Visualize recommendations by category for a user
    cb_recs = recommender.content_based_recommendation(example_user_id, n=20)
    if not cb_recs.empty and 'category' in cb_recs.columns:
        plt.figure(figsize=(10, 6))
        category_counts = cb_recs['category'].value_counts().head(10)
        sns.barplot(x=category_counts.values, y=category_counts.index)
        plt.title(f"Top Categories Recommended for User {example_user_id}")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig("category_recommendations.png")
        print("Saved category recommendations visualization")
    
    # Get evaluation metrics for different k values
    k_values = [5, 10, 15, 20]
    precision_values = []
    recall_values = []
    f1_values = []
    
    for k in k_values:
        print(f"Evaluating with k={k}...")
        metrics = evaluator.evaluate_recommendations(k=k)
        precision_values.append(metrics['precision'])
        recall_values.append(metrics['recall'])
        f1_values.append(metrics['f1'])
    
    # Plot metrics vs k
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, precision_values, 'o-', label='Precision')
    plt.plot(k_values, recall_values, 's-', label='Recall')
    plt.plot(k_values, f1_values, '^-', label='F1')
    plt.xlabel('k (number of recommendations)')
    plt.ylabel('Score')
    plt.title('Recommendation Performance vs. k')
    plt.legend()
    plt.grid(True)
    plt.savefig("metrics_vs_k.png")
    print("Saved metrics visualization")
    
    # Example 5: Compare different recommendation methods
    print("\nComparing recommendation methods...")
    methods = ['collaborative', 'content', 'hybrid', 'demographic']
    method_metrics = {}
    
    for method in methods:
        print(f"Evaluating {method} method...")
        precisions = []
        recalls = []
        
        # Evaluate for a subset of users
        for user_id in test_users[:20]:
            recommendations = recommender.recommend_for_user(user_id, method=method, n=10)
            
            if recommendations.empty:
                continue
                
            # Get actual ratings for this user
            with recommender.driver.session() as session:
                relevant_businesses = session.run("""
                    MATCH (u:User {user_id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
                    WHERE r.stars >= 4
                    RETURN b.business_id AS business_id
                """, user_id=user_id).values()
                
                relevant_businesses = set([item[0] for item in relevant_businesses])
                recommended_businesses = set(recommendations['business_id'].tolist())
                
                # Calculate relevant recommended items
                relevant_recommended = recommended_businesses.intersection(relevant_businesses)
                
                # Calculate precision and recall
                precision = len(relevant_recommended) / len(recommended_businesses) if recommended_businesses else 0
                recall = len(relevant_recommended) / len(relevant_businesses) if relevant_businesses else 0
                
                precisions.append(precision)
                recalls.append(recall)
        
        # Calculate average precision and recall
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        method_metrics[method] = {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': f1
        }
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    methods_list = list(method_metrics.keys())
    precision_list = [method_metrics[m]['precision'] for m in methods_list]
    recall_list = [method_metrics[m]['recall'] for m in methods_list]
    f1_list = [method_metrics[m]['f1'] for m in methods_list]
    
    x = np.arange(len(methods_list))
    width = 0.25
    
    plt.bar(x - width, precision_list, width, label='Precision')
    plt.bar(x, recall_list, width, label='Recall')
    plt.bar(x + width, f1_list, width, label='F1')
    
    plt.xlabel('Recommendation Method')
    plt.ylabel('Score')
    plt.title('Comparison of Recommendation Methods')
    plt.xticks(x, methods_list)
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig("method_comparison.png")
    print("Saved method comparison visualization")
    
    # Clean up
    recommender.close()
    print("\nDemo completed!")

if __name__ == "__main__":
    main()