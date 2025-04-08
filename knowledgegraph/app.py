import os
import pandas as pd
import argparse
from neo4j import GraphDatabase
from tabulate import tabulate

# Import the recommender class
from utils import YelpRecommender, RecommenderEvaluator, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

def display_dataframe(df):
    """Format and display a DataFrame nicely"""
    if len(df) == 0:
        print("No data to display.")
        return
    
    print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))

def get_user_recommendations(user_id, top_n=10):
    """Get recommendations for a user"""
    # Connect to the database
    recommender = YelpRecommender(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Load the trained model
        recommender.load_model()
        
        # Get recommendations
        recommendations = recommender.get_user_recommendations(user_id, top_n=top_n)
        
        if len(recommendations) == 0:
            print(f"No recommendations found for user {user_id}.")
            return
            
        # Format and display recommendations
        print(f"\nTop {len(recommendations)} recommendations for user {user_id}:")
        display_dataframe(recommendations[['name', 'predicted_rating', 'avg_rating', 'city']])
        
    finally:
        recommender.close()

def get_similar_businesses(business_id, top_n=10):
    """Get businesses similar to the specified business"""
    # Connect to the database
    recommender = YelpRecommender(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Load the trained model
        recommender.load_model()
        
        # Get business name
        with recommender.driver.session() as session:
            query = "MATCH (b:Business {business_id: $business_id}) RETURN b.name AS name"
            result = session.run(query, business_id=business_id)
            record = result.single()
            
            if record is None:
                print(f"Business with ID {business_id} not found.")
                return
                
            business_name = record['name']
        
        # Get similar businesses
        similar = recommender.get_similar_businesses(business_id, top_n=top_n)
        
        if len(similar) == 0:
            print(f"No similar businesses found for {business_name}.")
            return
            
        # Format and display similar businesses
        print(f"\nBusinesses similar to {business_name}:")
        display_dataframe(similar[['name', 'city', 'similarity', 'avg_rating']])
        
    finally:
        recommender.close()

def train_model(n_factors=50, limit=None):
    """Train a new recommendation model"""
    # Connect to the database
    recommender = YelpRecommender(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Train the model
        print(f"Training model with {n_factors} factors...")
        if limit:
            print(f"Using {limit} reviews for training.")
            
        recommender.train(n_factors=n_factors, limit=limit)
        
        # Save the model
        recommender.save_model()
        print("Model training completed and saved.")
        
    finally:
        recommender.close()

def evaluate_model():
    """Evaluate the recommendation model"""
    # Connect to the database
    evaluator = RecommenderEvaluator(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Fetch data for evaluation (limit for demonstration)
        print("Fetching data for evaluation...")
        data = evaluator._fetch_review_data(limit=50000)
        
        # Split into train and test sets
        train_data, test_data = evaluator.train_test_split_by_time(data)
        print(f"Train set size: {len(train_data)}")
        print(f"Test set size: {len(test_data)}")
        
        # Train a model on the training data
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
        n_factors = 50
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
        print("\nEvaluating model on test data:")
        metrics = evaluator.evaluate_model(recommender, test_data)
        
        # Display metrics
        print("\nEvaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
            
        print("\nEvaluation complete. Results saved to models/ratings_comparison.png")
        
    finally:
        evaluator.close()
        if 'recommender' in locals():
            recommender.close()

def find_random_users_and_businesses(n=5):
    """Find random users and businesses for testing"""
    # Connect to the database
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Find random users
            query = "MATCH (u:User) RETURN u.user_id AS user_id, u.name AS name LIMIT $limit"
            result = session.run(query, limit=n)
            users = [(record['user_id'], record['name']) for record in result]
            
            # Find random businesses
            query = "MATCH (b:Business) RETURN b.business_id AS business_id, b.name AS name LIMIT $limit"
            result = session.run(query, limit=n)
            businesses = [(record['business_id'], record['name']) for record in result]
            
            # Display the results
            print("\nRandom Users for Testing:")
            for user_id, name in users:
                print(f"User ID: {user_id}, Name: {name}")
                
            print("\nRandom Businesses for Testing:")
            for business_id, name in businesses:
                print(f"Business ID: {business_id}, Name: {name}")
                
    finally:
        driver.close()

def cross_validate():
    """Run cross-validation to find optimal model parameters"""
    # Connect to the database
    evaluator = RecommenderEvaluator(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Run cross-validation
        print("Running cross-validation to find optimal number of factors...")
        results, best_n_factors = evaluator.cross_validate(
            n_factors_list=[10, 20, 30, 50, 70, 100],
            limit=50000  # Limit data for faster execution
        )
        
        # Display results
        print("\nCross-Validation Results:")
        for n_factors, metrics in results.items():
            print(f"n_factors = {n_factors}: RMSE = {metrics['rmse']:.4f}, MAE = {metrics['mae']:.4f}")
        
        print(f"\nBest number of factors: {best_n_factors}")
        print("Cross-validation complete. Results saved to models/cross_validation_results.png")
        
        # Train final model with best parameters
        train_model(n_factors=best_n_factors)
        
    finally:
        evaluator.close()

def main():
    """Main function to parse arguments and run commands"""
    parser = argparse.ArgumentParser(description='Yelp Recommendation System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train recommendation model')
    train_parser.add_argument('--factors', type=int, default=50, help='Number of latent factors')
    train_parser.add_argument('--limit', type=int, help='Limit number of reviews for training')
    
    # Get recommendations command
    recommend_parser = subparsers.add_parser('recommend', help='Get recommendations for a user')
    recommend_parser.add_argument('user_id', help='User ID to get recommendations for')
    recommend_parser.add_argument('--top', type=int, default=10, help='Number of recommendations to return')
    
    # Get similar businesses command
    similar_parser = subparsers.add_parser('similar', help='Get businesses similar to a given business')
    similar_parser.add_argument('business_id', help='Business ID to find similar businesses for')
    similar_parser.add_argument('--top', type=int, default=10, help='Number of similar businesses to return')
    
    # Evaluate model command
    subparsers.add_parser('evaluate', help='Evaluate recommendation model')
    
    # Find random entities command
    subparsers.add_parser('random', help='Find random users and businesses')
    
    # Cross-validation command
    subparsers.add_parser('cv', help='Run cross-validation to find optimal parameters')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Execute command
    if args.command == 'train':
        train_model(n_factors=args.factors, limit=args.limit)
    elif args.command == 'recommend':
        get_user_recommendations(args.user_id, top_n=args.top)
    elif args.command == 'similar':
        get_similar_businesses(args.business_id, top_n=args.top)
    elif args.command == 'evaluate':
        evaluate_model()
    elif args.command == 'random':
        find_random_users_and_businesses()
    elif args.command == 'cv':
        cross_validate()
    else:
        parser.print_help()

if __name__ == "__main__":
    # Import packages for evaluate_model function
    import numpy as np
    from scipy.sparse.linalg import svds
    
    main()