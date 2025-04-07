import pandas as pd
import numpy as np
import pickle
import os
import json
from neo4j import GraphDatabase
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from collections import defaultdict

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
            """)
            
            ratings = []
            for record in result:
                ratings.append({
                    'user_id': record['user_id'],
                    'business_id': record['business_id'],
                    'stars': record['stars']
                })
            
            return pd.DataFrame(ratings)
    
    def build_user_preference_model(self, save_path='models'):
        """
        Build and save user preference models for faster recommendations
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
                
            # Build user-user similarities
            print("Building user similarity model...")
            user_similarity_result = session.run("""
                MATCH (u1:User)-[:WROTE]->(r1:Review)-[:ABOUT]->(b:Business)<-[:ABOUT]-(r2:Review)<-[:WROTE]-(u2:User)
                WHERE u1 <> u2
                WITH u1.user_id AS user1, u2.user_id AS user2, 
                     count(b) AS common_businesses,
                     sum(abs(r1.stars - r2.stars)) AS total_diff
                WHERE common_businesses >= 5
                RETURN user1, user2, common_businesses, 
                       total_diff / common_businesses AS avg_diff
                ORDER BY common_businesses DESC, avg_diff ASC
            """)
            
            user_similarities = defaultdict(dict)
            for record in user_similarity_result:
                user1 = record['user1']
                user2 = record['user2']
                common = record['common_businesses']
                avg_diff = record['avg_diff']
                
                # Calculate similarity score (higher is better)
                similarity = (1 / (1 + avg_diff)) * np.log1p(common)
                
                # Only store if similarity is above threshold
                if similarity > 0.2:  # Threshold to keep model size manageable
                    user_similarities[user1][user2] = similarity
            
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
            
            # If no pre-computed similarities, fetch from database
            if not similar_users:
                similar_users_result = session.run("""
                    MATCH (target:User {user_id: $user_id})-[:WROTE]->(r1:Review)-[:ABOUT]->(b:Business)
                    MATCH (other:User)-[:WROTE]->(r2:Review)-[:ABOUT]->(b)
                    WHERE target <> other
                    WITH other.user_id AS other_user_id, 
                         count(b) AS common_businesses,
                         abs(avg(r1.stars - r2.stars)) AS rating_diff
                    WHERE common_businesses > 5
                    RETURN other_user_id, common_businesses, rating_diff
                    ORDER BY rating_diff ASC, common_businesses DESC
                    LIMIT 50
                """, user_id=target_user_id)
                
                for record in similar_users_result:
                    similar_user_id = record['other_user_id']
                    common_businesses = record['common_businesses']
                    rating_diff = record['rating_diff']
                    
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
            
            for user_data in similar_users:
                similar_user_id = user_data['other_user_id']
                similarity = user_data['similarity']
                
                # Get highly-rated businesses from this similar user that target user hasn't rated
                similar_user_businesses = session.run("""
                    MATCH (u:User {user_id: $similar_user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
                    WHERE r.stars >= 4 AND NOT b.business_id IN $rated_businesses
                    RETURN b.business_id AS business_id, b.name AS name, b.stars AS avg_stars, 
                           r.stars AS user_stars, b.review_count AS review_count
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
            """, user_id=target_user_id).values()
            
            rated_businesses = [item[0] for item in rated_businesses]
            
            # Use pre-computed category preferences if available
            category_weights = {}
            if target_user_id in self.user_preferences:
                category_weights = self.user_preferences[target_user_id]
            
            # If no pre-computed preferences, fetch from database
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
            
            for category, weight in category_weights.items():
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
            
            # Find businesses highly rated by similar users that target user hasn't rated
            result = session.run("""
                MATCH (u:User)-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
                WHERE u.user_id IN $similar_users
                AND NOT b.business_id IN $rated_businesses
                AND r.stars >= 4
                WITH b, avg(r.stars) AS avg_user_rating, count(u) AS user_count
                WHERE user_count >= 2
                RETURN b.business_id AS business_id, b.name AS name,
                       b.stars AS avg_stars, b.review_count AS review_count,
                       avg_user_rating, user_count,
                       avg_user_rating * log10(user_count + 1) AS score
                ORDER BY score DESC
                LIMIT $limit
            """, similar_users=similar_user_ids, rated_businesses=rated_businesses, limit=n)
            
            recommendations = []
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
            
            return pd.DataFrame(recommendations)
    
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
        
    def create_train_test_split(self, test_ratio=0.2, min_ratings=10):
        """
        Split user ratings into training and testing datasets
        """
        # Fetch all ratings
        ratings_df = self.recommender.fetch_user_business_ratings()
        
        # Get users with sufficient number of ratings
        user_counts = ratings_df['user_id'].value_counts()
        eligible_users = user_counts[user_counts >= min_ratings].index.tolist()
        
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
    
    def evaluate_rmse(self, test_users, k=10):
        """
        Evaluate using Root Mean Squared Error
        Lower RMSE is better
        """
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
    
    def evaluate_precision_recall_at_k(self, test_users, k=10, threshold=4):
        """
        Evaluate using precision and recall at k
        """
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
    
    def map_at_k(self, test_users, k=10, threshold=4):
        """
        Calculate Mean Average Precision at k
        """
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
    
    def evaluate_recommendations(self, test_ratio=0.2, k=10, min_ratings=10):
        """
        Run a comprehensive evaluation of the recommendation system
        """
        # Create train/test split
        train_df, test_df, test_users = self.create_train_test_split(test_ratio, min_ratings)
        
        # Sample a subset of users for evaluation (for efficiency)
        if len(test_users) > 50:
            test_users = np.random.choice(test_users, 50, replace=False)
        
        # Calculate metrics
        rmse = self.evaluate_rmse(test_users, k)
        precision_recall = self.evaluate_precision_recall_at_k(test_users, k)
        map_score = self.map_at_k(test_users, k)
        
        results = {
            'rmse': rmse,
            'precision@k': precision_recall['precision'],
            'recall@k': precision_recall['recall'],
            'f1@k': precision_recall['f1'],
            'map@k': map_score
        }
        
        # Save evaluation results
        with open(os.path.join('models', 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        return results


# Example usage
def main():
    # Neo4j connection details
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password"  # Change to your actual password
    
    # Create recommendation system
    rec_system = YelpRecommendationSystem(uri, user, password)
    
    try:
        # Check if models exist and load them
        if os.path.exists(os.path.join('models', 'user_category_preferences.pkl')):
            print("Loading existing models...")
            rec_system.load_models()
        else:
            # Build and save models
            print("Building recommendation models...")
            rec_system.build_user_preference_model()
        
        # Get a sample user for demonstration
        with rec_system.driver.session() as session:
            sample_user = session.run("""
                MATCH (u:User)-[:WROTE]->(r:Review)
                WITH u, count(r) AS review_count
                WHERE review_count >= 10
                RETURN u.user_id AS user_id
                LIMIT 1
            """).single()
            
            if not sample_user:
                print("No suitable user found. Make sure the database is populated.")
                return
            
            user_id = sample_user['user_id']
            
            print(f"\nGenerating recommendations for user {user_id}:")
            
            # Get recommendations using different methods
            print("\n--- Collaborative Filtering Recommendations ---")
            cf_recs = rec_system.collaborative_filtering(user_id, n=5)
            print(cf_recs[['name', 'avg_stars', 'review_count', 'rec_score']])
            
            print("\n--- Content-Based Recommendations ---")
            cb_recs = rec_system.content_based_recommendation(user_id, n=5)
            print(cb_recs[['name', 'avg_stars', 'category', 'rec_score']])
            
            print("\n--- Hybrid Recommendations ---")
            hybrid_recs = rec_system.hybrid_recommendation(user_id, n=5)
            print(hybrid_recs[['name', 'avg_stars', 'review_count', 'score']])
            
            # Run evaluation
            print("\n--- Evaluating Recommendation System ---")
            evaluator = RecommendationEvaluation(rec_system)
            metrics = evaluator.evaluate_recommendations(test_ratio=0.2, k=10, min_ratings=5)
            
            print("\nEvaluation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            
            # Save model parameters after tuning (example)
            print("\n--- Updating Model Parameters ---")
            rec_system.update_parameters({
                'similarity_threshold': 0.25,  # Reduced threshold
                'cf_weight': 0.65,            # Increased collaborative filtering weight
                'cb_weight': 0.35             # Decreased content-based weight
            })
    
    finally:
        rec_system.close()

if __name__ == "__main__":
    main()