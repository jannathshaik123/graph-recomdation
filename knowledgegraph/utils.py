import os
import random
import numpy as np
import pandas as pd
import json
import pickle
from neo4j import GraphDatabase
from sklearn.model_selection import train_test_split
from datetime import datetime
from math import sqrt
from collections import defaultdict


NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

class YelpRecommendationSystem:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
    
    def get_user_ratings(self, user_id, limit=None):
        """Get all ratings (reviews) for a specific user"""
        with self.driver.session() as session:
            limit_clause = f"LIMIT {limit}" if limit else ""
            query = f"""
            MATCH (u:User {{user_id: $user_id}})-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
            RETURN b.business_id AS business_id, b.name AS business_name, r.stars AS rating, r.date AS date
            ORDER BY r.date DESC
            {limit_clause}
            """
            result = session.run(query, user_id=user_id)
            return [dict(record) for record in result]
    
    def get_business_details(self, business_id):
        """Get details about a specific business"""
        with self.driver.session() as session:
            query = """
            MATCH (b:Business {business_id: $business_id})
            OPTIONAL MATCH (b)-[:IN_CATEGORY]->(c:Category)
            OPTIONAL MATCH (b)-[:LOCATED_IN]->(city:City)
            RETURN b.business_id AS business_id,
                   b.name AS name,
                   b.stars AS avg_stars,
                   b.review_count AS review_count,
                   collect(DISTINCT c.name) AS categories,
                   city.name AS city
            """
            result = session.run(query, business_id=business_id)
            record = result.single()
            return dict(record) if record else None
    
    def get_similar_users(self, user_id, min_common=2, limit=50):
        """Find users with similar preferences based on review patterns"""
        with self.driver.session() as session:
            query = """
            MATCH (u1:User {user_id: $user_id})-[:WROTE]->(:Review)-[:ABOUT]->(b:Business)
            MATCH (u2:User)-[:WROTE]->(r2:Review)-[:ABOUT]->(b)
            WHERE u1 <> u2
            WITH u2, count(DISTINCT b) AS common_businesses
            WHERE common_businesses >= $min_common
            RETURN u2.user_id AS user_id, u2.name AS name, 
                   u2.average_stars AS avg_stars, common_businesses
            ORDER BY common_businesses DESC
            LIMIT $limit
            """
            result = session.run(query, user_id=user_id, min_common=min_common, limit=limit)
            return [dict(record) for record in result]
    
    def get_all_user_reviews(self, limit=None):
        """Get all user reviews for building training and test sets"""
        with self.driver.session() as session:
            limit_clause = f"LIMIT {limit}" if limit else ""
            query = f"""
            MATCH (u:User)-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
            RETURN u.user_id AS user_id, b.business_id AS business_id, 
                   r.stars AS rating, r.date AS date
            {limit_clause}
            """
            result = session.run(query)
            return [dict(record) for record in result]
    
    def get_user_average_rating(self, user_id):
        """Get the average rating for a user"""
        with self.driver.session() as session:
            query = """
            MATCH (u:User {user_id: $user_id})-[:WROTE]->(r:Review)
            RETURN avg(r.stars) AS avg_rating
            """
            result = session.run(query, user_id=user_id)
            record = result.single()
            return record["avg_rating"] if record and record["avg_rating"] is not None else 3.0

    def get_business_average_rating(self, business_id):
        """Get the average rating for a business"""
        with self.driver.session() as session:
            query = """
            MATCH (r:Review)-[:ABOUT]->(b:Business {business_id: $business_id})
            RETURN avg(r.stars) AS avg_rating
            """
            result = session.run(query, business_id=business_id)
            record = result.single()
            return record["avg_rating"] if record and record["avg_rating"] is not None else 3.0
    
    def get_rating(self, user_id, business_id):
        """Get a specific rating from a user for a business"""
        with self.driver.session() as session:
            query = """
            MATCH (u:User {user_id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business {business_id: $business_id})
            RETURN r.stars AS rating
            """
            result = session.run(query, user_id=user_id, business_id=business_id)
            record = result.single()
            return record["rating"] if record else None

    def collaborative_filtering_recommendations(self, user_id, k=10, top_n=10):
        """Get recommendations using user-based collaborative filtering"""
        
        user_reviews = self.get_user_ratings(user_id)
        if not user_reviews:
            return []
        
        
        user_businesses = {r['business_id']: r['rating'] for r in user_reviews}
        
        
        similar_users = self.get_similar_users(user_id, min_common=1, limit=k*3)
        
        
        recommendations = defaultdict(lambda: {'sum_sim': 0, 'weighted_sum': 0})
        
        with self.driver.session() as session:
            for sim_user in similar_users:
                sim_user_id = sim_user['user_id']
                
                
                query = """
                MATCH (u1:User {user_id: $user_id})-[:WROTE]->(r1:Review)-[:ABOUT]->(b:Business)
                MATCH (u2:User {user_id: $sim_user_id})-[:WROTE]->(r2:Review)-[:ABOUT]->(b)
                RETURN b.business_id AS business_id, r1.stars AS u1_rating, r2.stars AS u2_rating
                """
                result = session.run(query, user_id=user_id, sim_user_id=sim_user_id)
                
                common_ratings = [(r['business_id'], r['u1_rating'], r['u2_rating']) for r in result]
                
                if not common_ratings:
                    continue
                
                
                u1_ratings = [r[1] for r in common_ratings]
                u2_ratings = [r[2] for r in common_ratings]
                
                if len(u1_ratings) < 2:
                    continue
                
                try:
                    u1_mean = sum(u1_ratings) / len(u1_ratings)
                    u2_mean = sum(u2_ratings) / len(u2_ratings)
                    
                    numerator = sum((r1 - u1_mean) * (r2 - u2_mean) for r1, r2 in zip(u1_ratings, u2_ratings))
                    denominator1 = sqrt(sum((r1 - u1_mean) ** 2 for r1 in u1_ratings))
                    denominator2 = sqrt(sum((r2 - u2_mean) ** 2 for r2 in u2_ratings))
                    
                    
                    if denominator1 == 0 or denominator2 == 0:
                        similarity = 0
                    else:
                        similarity = numerator / (denominator1 * denominator2)
                    
                    
                    if similarity <= 0:
                        continue
                        
                    
                    query = """
                    MATCH (u:User {user_id: $sim_user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
                    WHERE NOT EXISTS {
                        MATCH (u2:User {user_id: $user_id})-[:WROTE]->(:Review)-[:ABOUT]->(b)
                    }
                    RETURN b.business_id AS business_id, r.stars AS rating
                    """
                    result = session.run(query, user_id=user_id, sim_user_id=sim_user_id)
                    
                    
                    for record in result:
                        business_id = record['business_id']
                        rating = record['rating']
                        
                        recommendations[business_id]['sum_sim'] += similarity
                        recommendations[business_id]['weighted_sum'] += similarity * rating
                    
                except Exception as e:
                    print(f"Error calculating similarity: {e}")
                    continue
        
        
        recommendation_list = []
        for business_id, values in recommendations.items():
            if values['sum_sim'] > 0:
                predicted_rating = values['weighted_sum'] / values['sum_sim']
                
                
                business_details = self.get_business_details(business_id)
                if business_details:
                    business_details['predicted_rating'] = predicted_rating
                    recommendation_list.append(business_details)
        
        
        recommendation_list.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return recommendation_list[:top_n]
    
    def content_based_recommendations(self, user_id, top_n=10):
        """Get recommendations based on business categories and user preferences"""
        with self.driver.session() as session:
            
            query = """
            MATCH (u:User {user_id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)-[:IN_CATEGORY]->(c:Category)
            WITH u, c, avg(r.stars) AS avg_rating
            WHERE avg_rating >= 3.5  // Focus on categories the user likes
            RETURN c.name AS category, avg_rating, count(*) AS frequency
            ORDER BY avg_rating DESC, frequency DESC
            """
            result = session.run(query, user_id=user_id)
            preferred_categories = [record['category'] for record in result]
            
            if not preferred_categories:
                return []
            
            
            top_categories = preferred_categories[:min(5, len(preferred_categories))]
            category_params = {f"category{i}": category for i, category in enumerate(top_categories)}
            
            category_conditions = " OR ".join([f"c.name = $category{i}" for i in range(len(top_categories))])
            
            query = f"""
            MATCH (b:Business)-[:IN_CATEGORY]->(c:Category)
            WHERE {category_conditions}
            AND NOT EXISTS {{
                MATCH (u:User {{user_id: $user_id}})-[:WROTE]->(:Review)-[:ABOUT]->(b)
            }}
            WITH b, collect(c.name) AS categories, b.stars AS avg_rating, b.review_count AS review_count
            RETURN b.business_id AS business_id, b.name AS name, categories, avg_rating, review_count
            ORDER BY avg_rating DESC, review_count DESC
            LIMIT $limit
            """
            
            params = {
                "user_id": user_id,
                "limit": top_n * 3
            }
            params.update(category_params)
            
            result = session.run(query, **params)
            recommendations = []
            
            for record in result:
                business = dict(record)
                
                
                category_match_score = sum(1 for cat in business['categories'] if cat in preferred_categories)
                business['score'] = (category_match_score * 0.7) + (business['avg_rating'] * 0.3)
                
                recommendations.append(business)
            
            
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:top_n]
    
    def hybrid_recommendations(self, user_id, top_n=10):
        """Combine collaborative filtering and content-based recommendations"""
        cf_recs = self.collaborative_filtering_recommendations(user_id, top_n=top_n)
        cb_recs = self.content_based_recommendations(user_id, top_n=top_n)
        
        
        all_recs = {}
        
        
        for rec in cf_recs:
            all_recs[rec['business_id']] = {
                'business_id': rec['business_id'],
                'name': rec['name'],
                'cf_score': rec['predicted_rating'],
                'cb_score': 0,
                'categories': rec['categories'],
                'city': rec['city'],
                'avg_stars': rec['avg_stars']
            }
        
        
        for rec in cb_recs:
            if rec['business_id'] in all_recs:
                all_recs[rec['business_id']]['cb_score'] = rec['score']
            else:
                all_recs[rec['business_id']] = {
                    'business_id': rec['business_id'],
                    'name': rec['name'],
                    'cf_score': 0,
                    'cb_score': rec['score'],
                    'categories': rec['categories'],
                    'city': None,  
                    'avg_stars': rec['avg_rating']
                }
        
        
        hybrid_recs = []
        for business_id, rec in all_recs.items():
            
            norm_cf = rec['cf_score'] / 5.0 if rec['cf_score'] > 0 else 0
            norm_cb = rec['cb_score'] / (1.0 + 5.0) if rec['cb_score'] > 0 else 0  
            
            
            hybrid_score = (norm_cf * 0.6) + (norm_cb * 0.4)
            
            rec['hybrid_score'] = hybrid_score
            rec['predicted_rating'] = hybrid_score * 5  
            hybrid_recs.append(rec)
        
        
        hybrid_recs.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return hybrid_recs[:top_n]
    
    def baseline_predict(self, user_id, business_id):
        """Baseline prediction model using global, user, and item biases"""
        
        with self.driver.session() as session:
            query = """
            MATCH ()-[:WROTE]->(r:Review)
            RETURN avg(r.stars) AS global_avg
            """
            result = session.run(query)
            record = result.single()
            global_avg = record["global_avg"] if record else 3.0
        
        
        user_avg = self.get_user_average_rating(user_id)
        user_bias = user_avg - global_avg
        
        
        business_avg = self.get_business_average_rating(business_id)
        business_bias = business_avg - global_avg
        
        
        predicted_rating = global_avg + user_bias + business_bias
        
        
        return max(1.0, min(5.0, predicted_rating))
    
    def matrix_factorization_predict(self, user_id, business_id, user_factors, business_factors):
        """Predict rating using pre-computed matrix factorization factors"""
        if user_id in user_factors and business_id in business_factors:
            
            prediction = np.dot(user_factors[user_id], business_factors[business_id])
            
            return max(1.0, min(5.0, prediction))
        else:
            
            return self.baseline_predict(user_id, business_id)
    
    def train_matrix_factorization(self, num_factors=20, learning_rate=0.005, regularization=0.02, num_iterations=50, sample_size=50000):
        """Train a matrix factorization model and return user and business factors"""
        print("Training matrix factorization model...")
        
        
        reviews = self.get_all_user_reviews(limit=sample_size)
        if not reviews:
            print("No reviews found for training.")
            return {}, {}
        
        
        users = set(review['user_id'] for review in reviews)
        businesses = set(review['business_id'] for review in reviews)
        
        
        user_to_idx = {user_id: i for i, user_id in enumerate(users)}
        business_to_idx = {business_id: i for i, business_id in enumerate(businesses)}
        
        
        num_users = len(users)
        num_businesses = len(businesses)
        
        
        np.random.seed(42)
        user_factors = np.random.normal(0, 0.1, (num_users, num_factors))
        business_factors = np.random.normal(0, 0.1, (num_businesses, num_factors))
        
        
        global_avg = sum(review['rating'] for review in reviews) / len(reviews)
        
        
        user_biases = np.zeros(num_users)
        business_biases = np.zeros(num_businesses)
        
        user_counts = defaultdict(int)
        business_counts = defaultdict(int)
        
        for review in reviews:
            user_idx = user_to_idx[review['user_id']]
            business_idx = business_to_idx[review['business_id']]
            user_biases[user_idx] += review['rating'] - global_avg
            business_biases[business_idx] += review['rating'] - global_avg
            user_counts[user_idx] += 1
            business_counts[business_idx] += 1
        
        for user_idx in range(num_users):
            count = user_counts[user_idx]
            if count > 0:
                user_biases[user_idx] /= count
        
        for business_idx in range(num_businesses):
            count = business_counts[business_idx]
            if count > 0:
                business_biases[business_idx] /= count
        
        
        for iteration in range(num_iterations):
            
            np.random.shuffle(reviews)
            
            total_error = 0
            
            for review in reviews:
                user_idx = user_to_idx[review['user_id']]
                business_idx = business_to_idx[review['business_id']]
                
                
                predicted = global_avg + user_biases[user_idx] + business_biases[business_idx] + \
                           np.dot(user_factors[user_idx], business_factors[business_idx])
                
                
                error = review['rating'] - predicted
                total_error += error ** 2
                
                
                user_biases[user_idx] += learning_rate * (error - regularization * user_biases[user_idx])
                business_biases[business_idx] += learning_rate * (error - regularization * business_biases[business_idx])
                
                
                user_factors_grad = error * business_factors[business_idx] - regularization * user_factors[user_idx]
                business_factors_grad = error * user_factors[user_idx] - regularization * business_factors[business_idx]
                
                user_factors[user_idx] += learning_rate * user_factors_grad
                business_factors[business_idx] += learning_rate * business_factors_grad
            
            rmse = sqrt(total_error / len(reviews))
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, RMSE: {rmse:.4f}")
        
        
        user_factors_dict = {user_id: user_factors[user_to_idx[user_id]] for user_id in users}
        business_factors_dict = {business_id: business_factors[business_to_idx[business_id]] for business_id in businesses}
        
        
        for user_id in users:
            user_idx = user_to_idx[user_id]
            user_factors_dict[user_id] = np.append(user_factors_dict[user_id], [user_biases[user_idx]])
        
        for business_id in businesses:
            business_idx = business_to_idx[business_id]
            business_factors_dict[business_id] = np.append(business_factors_dict[business_id], [business_biases[business_idx]])
        
        print("Matrix factorization training complete.")
        return user_factors_dict, business_factors_dict, global_avg

    def evaluate_recommendations(self, test_set, user_factors={}, business_factors={}, global_avg=3.0):
        """Evaluate recommendation algorithms using various metrics"""
        if not test_set:
            print("No test data available for evaluation.")
            return {}
        
        
        metrics = {
            'baseline': {'mae': 0, 'rmse': 0, 'count': 0},
            'matrix_factorization': {'mae': 0, 'rmse': 0, 'count': 0}
        }
        
        
        for record in test_set:
            user_id = record['user_id']
            business_id = record['business_id']
            actual_rating = record['rating']
            
            
            baseline_pred = self.baseline_predict(user_id, business_id)
            baseline_error = abs(actual_rating - baseline_pred)
            metrics['baseline']['mae'] += baseline_error
            metrics['baseline']['rmse'] += baseline_error ** 2
            metrics['baseline']['count'] += 1
            
            
            if user_factors and business_factors and user_id in user_factors and business_id in business_factors:
                user_vector = user_factors[user_id]
                business_vector = business_factors[business_id]
                
                
                user_bias = user_vector[-1]
                business_bias = business_vector[-1]
                
                
                dot_product = np.dot(user_vector[:-1], business_vector[:-1])
                
                
                mf_pred = global_avg + user_bias + business_bias + dot_product
                mf_pred = max(1.0, min(5.0, mf_pred))  
                
                mf_error = abs(actual_rating - mf_pred)
                metrics['matrix_factorization']['mae'] += mf_error
                metrics['matrix_factorization']['rmse'] += mf_error ** 2
                metrics['matrix_factorization']['count'] += 1
        
        
        for model in metrics:
            count = metrics[model]['count']
            if count > 0:
                metrics[model]['mae'] /= count
                metrics[model]['rmse'] = sqrt(metrics[model]['rmse'] / count)
                print(f"{model.capitalize()} - MAE: {metrics[model]['mae']:.4f}, RMSE: {metrics[model]['rmse']:.4f}")
            else:
                print(f"No valid predictions for {model} model.")
        
        return metrics

    def split_train_test(self, test_size=0.2, min_reviews=5, sample_size=100000):
        """Split data into training and test sets"""
        print("Splitting data into training and test sets...")
        
        
        all_reviews = self.get_all_user_reviews(limit=sample_size)
        if not all_reviews:
            print("No reviews found.")
            return [], []
        
        
        df = pd.DataFrame(all_reviews)
        
        
        df['date'] = pd.to_datetime(df['date'])
        
        
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_reviews].index.tolist()
        
        
        df_valid = df[df['user_id'].isin(valid_users)]
        
        
        df_valid = df_valid.sort_values(['user_id', 'date'])
        
        
        grouped = df_valid.groupby('user_id')
        
        train_set = []
        test_set = []
        
        
        for user_id, group in grouped:
            n_reviews = len(group)
            n_test = max(1, int(n_reviews * test_size))
            
            user_train = group.iloc[:-n_test].to_dict('records')
            user_test = group.iloc[-n_test:].to_dict('records')
            
            train_set.extend(user_train)
            test_set.extend(user_test)
        
        print(f"Train set: {len(train_set)} reviews, Test set: {len(test_set)} reviews")
        return train_set, test_set
    
    def save_matrix_factorization_model(self, user_factors, business_factors, global_avg, model_dir="models"):
        """Save trained matrix factorization model to files"""
        
        os.makedirs(model_dir, exist_ok=True)
        
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        
        user_factors_file = os.path.join(model_dir, f"user_factors_{timestamp}.pkl")
        with open(user_factors_file, 'wb') as f:
            pickle.dump(user_factors, f)
        
        
        business_factors_file = os.path.join(model_dir, f"business_factors_{timestamp}.pkl")
        with open(business_factors_file, 'wb') as f:
            pickle.dump(business_factors, f)
        
        
        metadata = {
            "global_avg": global_avg,
            "timestamp": timestamp,
            "num_users": len(user_factors),
            "num_businesses": len(business_factors),
            "factors_shape": next(iter(user_factors.values())).shape[0] - 1  
        }
        
        metadata_file = os.path.join(model_dir, f"mf_metadata_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        
        latest_info = {
            "timestamp": timestamp,
            "user_factors_file": user_factors_file,
            "business_factors_file": business_factors_file,
            "metadata_file": metadata_file
        }
        
        latest_file = os.path.join(model_dir, "latest_model.json")
        with open(latest_file, 'w') as f:
            json.dump(latest_info, f)
        
        print(f"Model saved successfully to {model_dir} directory with timestamp {timestamp}")
        return latest_info

    def load_matrix_factorization_model(self, model_dir="models", timestamp=None):
        """Load trained matrix factorization model from files"""
        
        if not os.path.exists(model_dir):
            print(f"Models directory {model_dir} not found.")
            return None, None, None
        
        try:
            
            if timestamp is None:
                latest_file = os.path.join(model_dir, "latest_model.json")
                if not os.path.exists(latest_file):
                    print("No saved model found.")
                    return None, None, None
                
                with open(latest_file, 'r') as f:
                    latest_info = json.load(f)
                
                timestamp = latest_info["timestamp"]
                user_factors_file = latest_info["user_factors_file"]
                business_factors_file = latest_info["business_factors_file"]
                metadata_file = latest_info["metadata_file"]
            else:
                
                user_factors_file = os.path.join(model_dir, f"user_factors_{timestamp}.pkl")
                business_factors_file = os.path.join(model_dir, f"business_factors_{timestamp}.pkl")
                metadata_file = os.path.join(model_dir, f"mf_metadata_{timestamp}.json")
            
            
            with open(user_factors_file, 'rb') as f:
                user_factors = pickle.load(f)
            
            
            with open(business_factors_file, 'rb') as f:
                business_factors = pickle.load(f)
            
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            global_avg = metadata["global_avg"]
            
            print(f"Model loaded successfully from {model_dir} with timestamp {timestamp}")
            print(f"Model contains {metadata['num_users']} users and {metadata['num_businesses']} businesses")
            return user_factors, business_factors, global_avg
        
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None, None

    def get_available_models(self, model_dir="models"):
        """Get list of available trained models"""
        if not os.path.exists(model_dir):
            print(f"Models directory {model_dir} not found.")
            return []
        
        
        metadata_files = [f for f in os.listdir(model_dir) if f.startswith("mf_metadata_")]
        models = []
        
        for file in metadata_files:
            timestamp = file.replace("mf_metadata_", "").replace(".json", "")
            metadata_path = os.path.join(model_dir, file)
            
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                models.append({
                    "timestamp": timestamp,
                    "num_users": metadata.get("num_users", "unknown"),
                    "num_businesses": metadata.get("num_businesses", "unknown"),
                    "global_avg": metadata.get("global_avg", "unknown"),
                    "factors_shape": metadata.get("factors_shape", "unknown")
                })
            except Exception as e:
                print(f"Error reading metadata for {file}: {e}")
        
        
        models.sort(key=lambda x: x["timestamp"], reverse=True)
        return models

def main():
    
    recommender = YelpRecommendationSystem(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        
        available_models = recommender.get_available_models()
        
        user_factors = None
        business_factors = None
        global_avg = None
        
        if available_models:
            print("\nAvailable models:")
            for i, model in enumerate(available_models):
                print(f"{i+1}. Model from {model['timestamp']} - {model['num_users']} users, {model['num_businesses']} businesses")
            
            
            print("\nOptions:")
            print("1. Load the latest model")
            print("2. Train a new model")
            choice = input("Enter your choice (1 or 2): ")
            
            if choice == "1":
                
                user_factors, business_factors, global_avg = recommender.load_matrix_factorization_model()
            else:
                print("Training new model...")
        else:
            print("No saved models found. Training new model...")
        
        
        if user_factors is None:
            
            train_set, test_set = recommender.split_train_test(test_size=0.2, min_reviews=3, sample_size=50000)
            
            if not train_set or not test_set:
                print("Insufficient data for evaluation.")
                return
            
            
            user_factors, business_factors, global_avg = recommender.train_matrix_factorization(
                num_factors=15,
                learning_rate=0.005,
                regularization=0.02,
                num_iterations=20,
                sample_size=len(train_set)
            )
            
            
            recommender.save_matrix_factorization_model(user_factors, business_factors, global_avg)
        else:
            
            
            _, test_set = recommender.split_train_test(test_size=0.2, min_reviews=3, sample_size=10000)
        
        
        print("\nEvaluating recommendation algorithms...")
        metrics = recommender.evaluate_recommendations(test_set, user_factors, business_factors, global_avg)
        
        
        example_user_id = test_set[0]['user_id']  
        print(f"\nExample recommendations for user {example_user_id}:")
        
        
        print("\nCollaborative Filtering Recommendations:")
        cf_recs = recommender.collaborative_filtering_recommendations(example_user_id, top_n=5)
        for i, rec in enumerate(cf_recs):
            print(f"{i+1}. {rec['name']} - Predicted Rating: {rec['predicted_rating']:.2f} - Categories: {', '.join(rec['categories'][:3])}")
        
        
        print("\nContent-Based Recommendations:")
        cb_recs = recommender.content_based_recommendations(example_user_id, top_n=5)
        for i, rec in enumerate(cb_recs):
            print(f"{i+1}. {rec['name']} - Score: {rec['score']:.2f} - Categories: {', '.join(rec['categories'][:3])}")
        
        
        print("\nHybrid Recommendations:")
        hybrid_recs = recommender.hybrid_recommendations(example_user_id, top_n=5)
        for i, rec in enumerate(hybrid_recs):
            print(f"{i+1}. {rec['name']} - Predicted Rating: {rec['predicted_rating']:.2f} - Categories: {', '.join(rec['categories'][:3])}")
        
    finally:
        recommender.close()

if __name__ == "__main__":
    main()
    