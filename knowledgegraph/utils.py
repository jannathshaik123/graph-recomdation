import os
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
import warnings
warnings.filterwarnings('ignore')

# Neo4j connection configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # Change this to your actual password

class YelpRecommender:
    def __init__(self, uri, user, password):
        """Initialize the recommender with Neo4j connection"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.user_mapping = {}  # Maps user_id to matrix index
        self.business_mapping = {}  # Maps business_id to matrix index
        self.reverse_user_mapping = {}  # Maps matrix index to user_id
        self.reverse_business_mapping = {}  # Maps matrix index to business_id
        self.model = None
        self.global_average = None
        
    def close(self):
        """Close the Neo4j connection"""
        self.driver.close()
        
    def _fetch_data(self, limit=None):
        """Fetch review data from Neo4j"""
        print("Fetching review data from Neo4j...")
        
        with self.driver.session() as session:
            # Query to get all review data
            query = """
            MATCH (u:User)-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
            RETURN u.user_id AS user_id, b.business_id AS business_id, 
                   r.stars AS rating, r.date AS date
            ORDER BY r.date
            """
            
            if limit:
                query += f" LIMIT {limit}"
                
            result = session.run(query)
            
            # Convert to DataFrame
            records = [record for record in result]
            df = pd.DataFrame(records)
            
            print(f"Fetched {len(df)} reviews")
            return df
    
    def _fetch_business_info(self, business_ids):
        """Fetch business information for a list of business IDs"""
        with self.driver.session() as session:
            query = """
            MATCH (b:Business)
            WHERE b.business_id IN $business_ids
            RETURN b.business_id AS business_id, b.name AS name, 
                   b.stars AS avg_rating, b.review_count AS review_count,
                   b.city AS city
            """
            result = session.run(query, business_ids=list(business_ids))
            records = [record for record in result]
            return pd.DataFrame(records)
            
    def _create_mappings(self, df):
        """Create mapping dictionaries for users and businesses"""
        print("Creating user and business mappings...")
        
        # Create user mapping
        unique_users = df['user_id'].unique()
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.reverse_user_mapping = {idx: user_id for user_id, idx in self.user_mapping.items()}
        
        # Create business mapping
        unique_businesses = df['business_id'].unique()
        self.business_mapping = {biz_id: idx for idx, biz_id in enumerate(unique_businesses)}
        self.reverse_business_mapping = {idx: biz_id for biz_id, idx in self.business_mapping.items()}
        
        print(f"Created mappings for {len(unique_users)} users and {len(unique_businesses)} businesses")
    
    def _create_ratings_matrix(self, df):
        """Create the user-item ratings matrix"""
        print("Creating user-item ratings matrix...")
        
        num_users = len(self.user_mapping)
        num_businesses = len(self.business_mapping)
        
        # Initialize ratings matrix with zeros
        ratings_matrix = np.zeros((num_users, num_businesses))
        
        # Fill the matrix with ratings
        for _, row in df.iterrows():
            user_idx = self.user_mapping[row['user_id']]
            business_idx = self.business_mapping[row['business_id']]
            ratings_matrix[user_idx, business_idx] = row['rating']
        
        # Calculate global average for unrated items
        self.global_average = df['rating'].mean()
        
        print(f"Created ratings matrix of shape {ratings_matrix.shape}")
        return ratings_matrix
    
    def train(self, n_factors=50, limit=None):
        """Train the collaborative filtering model with SVD"""
        print("Training recommendation model...")
        
        # Fetch data
        df = self._fetch_data(limit=limit)
        
        # Create mappings
        self._create_mappings(df)
        
        # Create ratings matrix
        ratings_matrix = self._create_ratings_matrix(df)
        
        # Calculate user and item biases
        user_ratings_mean = np.nanmean(ratings_matrix, axis=1).reshape(-1, 1)
        ratings_demeaned = ratings_matrix - user_ratings_mean
        
        # Perform SVD
        U, sigma, Vt = svds(ratings_demeaned, k=n_factors)
        
        # Convert sigma to diagonal matrix
        sigma_diag = np.diag(sigma)
        
        # Store the model components
        self.model = {
            'U': U,
            'sigma': sigma_diag,
            'Vt': Vt,
            'user_ratings_mean': user_ratings_mean,
            'global_average': self.global_average
        }
        
        print("Model training complete")
        return self
    
    def save_model(self, folder_path="models"):
        """Save the trained model to disk"""
        # Create models directory if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        
        # Save model components
        model_data = {
            'model': self.model,
            'user_mapping': self.user_mapping,
            'business_mapping': self.business_mapping,
            'reverse_user_mapping': self.reverse_user_mapping,
            'reverse_business_mapping': self.reverse_business_mapping,
            'global_average': self.global_average
        }
        
        model_path = os.path.join(folder_path, "yelp_recommender.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Model saved to {model_path}")
    
    def load_model(self, folder_path="models"):
        """Load a trained model from disk"""
        model_path = os.path.join(folder_path, "yelp_recommender.pkl")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.user_mapping = model_data['user_mapping']
        self.business_mapping = model_data['business_mapping']
        self.reverse_user_mapping = model_data['reverse_user_mapping']
        self.reverse_business_mapping = model_data['reverse_business_mapping']
        self.global_average = model_data['global_average']
        
        print(f"Model loaded from {model_path}")
        return self
    
    def predict_rating(self, user_id, business_id):
        """Predict rating for a specific user and business"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
            
        if user_id not in self.user_mapping or business_id not in self.business_mapping:
            # Return global average if user or business not in training data
            return self.global_average
            
        user_idx = self.user_mapping[user_id]
        business_idx = self.business_mapping[business_id]
        
        # Get model components
        U = self.model['U']
        sigma = self.model['sigma']
        Vt = self.model['Vt']
        user_ratings_mean = self.model['user_ratings_mean']
        
        # Predict rating
        pred = user_ratings_mean[user_idx] + np.dot(np.dot(U[user_idx, :], sigma), Vt[:, business_idx])
        
        # Clip prediction to valid rating range (1-5)
        return min(max(pred, 1.0), 5.0)
    
    def get_user_recommendations(self, user_id, top_n=10, filter_rated=True):
        """Get top N recommendations for a user"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
            
        if user_id not in self.user_mapping:
            print(f"User {user_id} not found in training data.")
            return pd.DataFrame()
            
        user_idx = self.user_mapping[user_id]
        
        # Get model components
        U = self.model['U']
        sigma = self.model['sigma']
        Vt = self.model['Vt']
        user_ratings_mean = self.model['user_ratings_mean']
        
        # Get predictions for all businesses
        user_pred_ratings = user_ratings_mean[user_idx] + np.dot(np.dot(U[user_idx, :], sigma), Vt)
        
        # Convert to DataFrame
        user_pred_df = pd.DataFrame({
            'business_id': [self.reverse_business_mapping[i] for i in range(len(user_pred_ratings))],
            'predicted_rating': user_pred_ratings
        })
        
        # Filter out businesses the user has already rated if requested
        if filter_rated:
            with self.driver.session() as session:
                query = """
                MATCH (u:User {user_id: $user_id})-[:WROTE]->(:Review)-[:ABOUT]->(b:Business)
                RETURN b.business_id AS business_id
                """
                result = session.run(query, user_id=user_id)
                rated_businesses = [record['business_id'] for record in result]
                user_pred_df = user_pred_df[~user_pred_df['business_id'].isin(rated_businesses)]
        
        # Sort by predicted rating and get top N
        top_recommendations = user_pred_df.sort_values('predicted_rating', ascending=False).head(top_n)
        
        # Fetch additional information about the recommended businesses
        business_info = self._fetch_business_info(top_recommendations['business_id'])
        
        # Merge with prediction data
        recommendations = pd.merge(top_recommendations, business_info, on='business_id')
        
        return recommendations[['business_id', 'name', 'predicted_rating', 'avg_rating', 'review_count', 'city']]
    
    def get_similar_users(self, user_id, top_n=10):
        """Find users similar to the given user based on latent factors"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
            
        if user_id not in self.user_mapping:
            print(f"User {user_id} not found in training data.")
            return []
            
        user_idx = self.user_mapping[user_id]
        
        # Get user latent factors
        user_factors = self.model['U'][user_idx, :]
        
        # Calculate cosine similarity with all other users
        similarities = []
        for idx, other_user_id in self.reverse_user_mapping.items():
            if idx == user_idx:
                continue
                
            other_factors = self.model['U'][idx, :]
            similarity = np.dot(user_factors, other_factors) / (np.linalg.norm(user_factors) * np.linalg.norm(other_factors))
            similarities.append((other_user_id, similarity))
        
        # Sort by similarity and return top N
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    
    def get_similar_businesses(self, business_id, top_n=10):
        """Find businesses similar to the given business based on latent factors"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
            
        if business_id not in self.business_mapping:
            print(f"Business {business_id} not found in training data.")
            return []
            
        business_idx = self.business_mapping[business_id]
        
        # Get business latent factors
        business_factors = self.model['Vt'][:, business_idx]
        
        # Calculate cosine similarity with all other businesses
        similarities = []
        for idx, other_business_id in self.reverse_business_mapping.items():
            if idx == business_idx:
                continue
                
            other_factors = self.model['Vt'][:, idx]
            similarity = np.dot(business_factors, other_factors) / (np.linalg.norm(business_factors) * np.linalg.norm(other_factors))
            similarities.append((other_business_id, similarity))
        
        # Sort by similarity and return top N
        top_similar = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
        
        # Get business information
        business_ids = [item[0] for item in top_similar]
        business_info = self._fetch_business_info(business_ids)
        
        # Add similarity scores
        similarity_dict = {item[0]: item[1] for item in top_similar}
        business_info['similarity'] = business_info['business_id'].map(similarity_dict)
        
        return business_info.sort_values('similarity', ascending=False)

class RecommenderEvaluator:
    def __init__(self, uri, user, password):
        """Initialize the evaluator with Neo4j connection"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.recommender = None
    
    def close(self):
        """Close the Neo4j connection"""
        self.driver.close()
        
    def _fetch_review_data(self, limit=None):
        """Fetch review data from Neo4j"""
        with self.driver.session() as session:
            query = """
            MATCH (u:User)-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
            RETURN u.user_id AS user_id, b.business_id AS business_id, 
                   r.stars AS rating, r.date AS date
            ORDER BY r.date
            """
            
            if limit:
                query += f" LIMIT {limit}"
                
            result = session.run(query)
            records = [record for record in result]
            return pd.DataFrame(records)
    
    def train_test_split_by_time(self, data=None, test_size=0.2, date_column='date'):
        """Split data into train and test sets based on time"""
        if data is None:
            data = self._fetch_review_data()
            
        # Convert date string to datetime if needed
        if isinstance(data[date_column].iloc[0], str):
            data[date_column] = pd.to_datetime(data[date_column])
            
        # Sort by date
        data = data.sort_values(by=date_column)
        
        # Determine split point
        split_idx = int(len(data) * (1 - test_size))
        
        # Split data
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        return train_data, test_data
    
    def evaluate_model(self, recommender, test_data, k_values=[5, 10, 20]):
        """Evaluate the recommender model using various metrics"""
        self.recommender = recommender
        
        # Calculate prediction metrics
        print("Calculating prediction metrics...")
        true_ratings = []
        predicted_ratings = []
        
        # Predict ratings for test data
        for _, row in test_data.iterrows():
            user_id = row['user_id']
            business_id = row['business_id']
            true_rating = row['rating']
            
            # Skip if user or business not in training data
            if user_id not in recommender.user_mapping or business_id not in recommender.business_mapping:
                continue
                
            predicted_rating = recommender.predict_rating(user_id, business_id)
            
            true_ratings.append(true_rating)
            predicted_ratings.append(predicted_rating)
        
        # Ensure we have predictions to evaluate
        if len(true_ratings) == 0:
            print("No predictions available for evaluation")
            return {}
            
        # Calculate error metrics
        mae = mean_absolute_error(true_ratings, predicted_ratings)
        rmse = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
        r2 = r2_score(true_ratings, predicted_ratings)
        
        # Calculate average difference to ensure RMSE isn't zero or infinite
        avg_diff = np.mean(np.abs(np.array(true_ratings) - np.array(predicted_ratings)))
        
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"Average difference: {avg_diff:.4f}")
        
        # Plot actual vs predicted ratings
        plt.figure(figsize=(10, 6))
        plt.scatter(true_ratings, predicted_ratings, alpha=0.5)
        plt.plot([1, 5], [1, 5], 'r--')
        plt.xlabel('Actual Ratings')
        plt.ylabel('Predicted Ratings')
        plt.title('Actual vs Predicted Ratings')
        plt.axis('equal')
        plt.axis([1, 5, 1, 5])
        plt.savefig("models/ratings_comparison.png")
        plt.close()
        
        # Return metrics
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'avg_diff': avg_diff,
            'num_predictions': len(true_ratings)
        }
        
        return metrics
    
    def cross_validate(self, n_factors_list=[10, 20, 50, 100], limit=None):
        """Perform cross-validation to find optimal number of factors"""
        print("Performing cross-validation...")
        
        # Fetch data
        data = self._fetch_review_data(limit=limit)
        
        # Split data into train and test sets
        train_data, test_data = self.train_test_split_by_time(data)
        
        # Initialize results dictionary
        results = {}
        
        # Test different numbers of factors
        for n_factors in n_factors_list:
            print(f"\nTraining model with {n_factors} factors...")
            
            # Create and train recommender
            recommender = YelpRecommender(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            
            # Create mappings and train model
            recommender._create_mappings(train_data)
            
            # Create ratings matrix
            ratings_matrix = np.zeros((len(recommender.user_mapping), len(recommender.business_mapping)))
            
            # Fill matrix with ratings
            for _, row in train_data.iterrows():
                user_id = row['user_id']
                business_id = row['business_id']
                
                if user_id in recommender.user_mapping and business_id in recommender.business_mapping:
                    user_idx = recommender.user_mapping[user_id]
                    business_idx = recommender.business_mapping[business_id]
                    ratings_matrix[user_idx, business_idx] = row['rating']
            
            # Calculate global average
            recommender.global_average = train_data['rating'].mean()
            
            # Calculate user means
            user_ratings_mean = np.nanmean(ratings_matrix, axis=1).reshape(-1, 1)
            ratings_demeaned = ratings_matrix - user_ratings_mean
            
            # Perform SVD
            U, sigma, Vt = svds(ratings_demeaned, k=n_factors)
            sigma_diag = np.diag(sigma)
            
            # Store model
            recommender.model = {
                'U': U,
                'sigma': sigma_diag,
                'Vt': Vt,
                'user_ratings_mean': user_ratings_mean,
                'global_average': recommender.global_average
            }
            
            # Evaluate model
            metrics = self.evaluate_model(recommender, test_data)
            results[n_factors] = metrics
            
            # Close connection
            recommender.close()
        
        # Plot results
        plt.figure(figsize=(12, 6))
        
        # Plot RMSE
        plt.subplot(1, 2, 1)
        plt.plot(list(results.keys()), [results[k]['rmse'] for k in results.keys()], 'o-')
        plt.title('RMSE vs Number of Factors')
        plt.xlabel('Number of Factors')
        plt.ylabel('RMSE')
        plt.grid(True)
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(list(results.keys()), [results[k]['mae'] for k in results.keys()], 'o-')
        plt.title('MAE vs Number of Factors')
        plt.xlabel('Number of Factors')
        plt.ylabel('MAE')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("models/cross_validation_results.png")
        plt.close()
        
        # Find best number of factors
        best_n_factors = min(results.keys(), key=lambda k: results[k]['rmse'])
        print(f"\nBest number of factors: {best_n_factors} (RMSE: {results[best_n_factors]['rmse']:.4f})")
        
        return results, best_n_factors

# Sample usage
def sample_usage():
    """Demonstrate usage of the recommender system"""
    # Connect to Neo4j
    recommender = YelpRecommender(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # Train model (limit to 10000 reviews for faster execution)
    recommender.train(n_factors=50, limit=10000)
    
    # Save model
    recommender.save_model()
    
    # Get recommendations for a random user
    with recommender.driver.session() as session:
        # Get a random user ID
        query = "MATCH (u:User) RETURN u.user_id AS user_id LIMIT 1"
        result = session.run(query)
        user_id = result.single()['user_id']
    
    print(f"\nGetting recommendations for user {user_id}:")
    recommendations = recommender.get_user_recommendations(user_id, top_n=5)
    print(recommendations)
    
    # Get a random business for similarity search
    with recommender.driver.session() as session:
        query = "MATCH (b:Business) RETURN b.business_id AS business_id, b.name AS name LIMIT 1"
        result = session.run(query)
        record = result.single()
        business_id = record['business_id']
        business_name = record['name']
    
    print(f"\nFinding businesses similar to {business_name} ({business_id}):")
    similar_businesses = recommender.get_similar_businesses(business_id, top_n=5)
    print(similar_businesses[['name', 'city', 'similarity']])
    
    # Perform evaluation
    evaluator = RecommenderEvaluator(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # Fetch data for evaluation
    data = evaluator._fetch_review_data(limit=20000)
    train_data, test_data = evaluator.train_test_split_by_time(data)
    
    print(f"\nTraining data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")
    
    # Create a new recommender for training
    eval_recommender = YelpRecommender(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # Create mappings and train model on training data
    eval_recommender._create_mappings(train_data)
    
    # Create ratings matrix
    ratings_matrix = np.zeros((len(eval_recommender.user_mapping), len(eval_recommender.business_mapping)))
    
    # Fill matrix with ratings
    for _, row in train_data.iterrows():
        user_id = row['user_id']
        business_id = row['business_id']
        
        if user_id in eval_recommender.user_mapping and business_id in eval_recommender.business_mapping:
            user_idx = eval_recommender.user_mapping[user_id]
            business_idx = eval_recommender.business_mapping[business_id]
            ratings_matrix[user_idx, business_idx] = row['rating']
    
    # Calculate global average
    eval_recommender.global_average = train_data['rating'].mean()
    
    # Calculate user means
    user_ratings_mean = np.nanmean(ratings_matrix, axis=1).reshape(-1, 1)
    ratings_demeaned = ratings_matrix - user_ratings_mean
    
    # Perform SVD
    n_factors = 50
    U, sigma, Vt = svds(ratings_demeaned, k=n_factors)
    sigma_diag = np.diag(sigma)
    
    # Store model
    eval_recommender.model = {
        'U': U,
        'sigma': sigma_diag,
        'Vt': Vt,
        'user_ratings_mean': user_ratings_mean,
        'global_average': eval_recommender.global_average
    }
    
    # Evaluate model
    print("\nEvaluating model on test data:")
    metrics = evaluator.evaluate_model(eval_recommender, test_data)
    
    # Clean up
    recommender.close()
    eval_recommender.close()
    evaluator.close()
    
    return metrics

# Run cross-validation to find optimal parameters
def run_cross_validation():
    """Run cross-validation to find optimal model parameters"""
    evaluator = RecommenderEvaluator(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # Run cross-validation with different numbers of factors
    results, best_n_factors = evaluator.cross_validate(
        n_factors_list=[10, 20, 30, 50, 70, 100],
        limit=50000  # Limit data for faster execution
    )
    
    # Train final model with best number of factors
    print(f"\nTraining final model with {best_n_factors} factors...")
    recommender = YelpRecommender(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    recommender.train(n_factors=best_n_factors)
    
    # Save final model
    recommender.save_model()
    
    # Clean up
    recommender.close()
    evaluator.close()
    
    return best_n_factors, results

if __name__ == "__main__":
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Run sample usage
    print("Running sample usage...")
    metrics = sample_usage()
    
    # Optionally run cross-validation (uncomment to run)
    # print("\nRunning cross-validation...")
    # best_n_factors, cv_results = run_cross_validation()