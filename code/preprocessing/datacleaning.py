import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# Define paths to the Yelp dataset
base_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data")
reviews_path = os.path.join(base_path, "yelp_training_set\yelp_training_set_review.json")
business_path = os.path.join(base_path, "yelp_training_set\yelp_training_set_business.json")
user_path = os.path.join(base_path, "yelp_training_set\yelp_training_set_user.json")

# Function to load JSON data
def load_json_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
    return pd.DataFrame(data)

# Load datasets
print("Loading datasets...")
reviews_df = load_json_data(reviews_path)
business_df = load_json_data(business_path)
users_df = load_json_data(user_path)

print("Initial dataset shapes:")
print(f"Reviews: {reviews_df.shape}")
print(f"Businesses: {business_df.shape}")
print(f"Users: {users_df.shape}")

# Data Cleaning
def clean_reviews(df):
    """Clean the reviews dataset"""
    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates
    cleaned_df = cleaned_df.drop_duplicates(subset=['review_id'])
    
    # Convert date to datetime
    cleaned_df['date'] = pd.to_datetime(cleaned_df['date'], errors='coerce')
    
    # Extract vote counts
    cleaned_df['useful_votes'] = cleaned_df['votes'].apply(lambda x: x.get('useful', 0) if isinstance(x, dict) else 0)
    cleaned_df['funny_votes'] = cleaned_df['votes'].apply(lambda x: x.get('funny', 0) if isinstance(x, dict) else 0)
    cleaned_df['cool_votes'] = cleaned_df['votes'].apply(lambda x: x.get('cool', 0) if isinstance(x, dict) else 0)
    
    # Handle missing values
    cleaned_df['text'] = cleaned_df['text'].fillna('')
    cleaned_df['stars'] = cleaned_df['stars'].fillna(cleaned_df['stars'].median())
    
    # Remove reviews with invalid business_id or user_id
    cleaned_df = cleaned_df.dropna(subset=['business_id', 'user_id'])
    
    return cleaned_df

def clean_businesses(df):
    """Clean the businesses dataset"""
    cleaned_df = df.copy()
    
    # Remove duplicates
    cleaned_df = cleaned_df.drop_duplicates(subset=['business_id'])
    
    # Handle missing values
    cleaned_df['name'] = cleaned_df['name'].fillna('Unknown')
    cleaned_df['city'] = cleaned_df['city'].fillna('Unknown')
    cleaned_df['state'] = cleaned_df['state'].fillna('Unknown')
    cleaned_df['stars'] = cleaned_df['stars'].fillna(cleaned_df['stars'].median())
    cleaned_df['review_count'] = cleaned_df['review_count'].fillna(0)
    cleaned_df['categories'] = cleaned_df['categories'].fillna('')
    
    # Convert categories to list format if it's a string
    cleaned_df['categories'] = cleaned_df['categories'].apply(
        lambda x: [cat.strip() for cat in x.split(',')] if isinstance(x, str) and x else []
    )
    
    # Handle open status
    cleaned_df['open'] = cleaned_df['open'].fillna(True)
    
    return cleaned_df

def clean_users(df):
    """Clean the users dataset"""
    cleaned_df = df.copy()
    
    # Remove duplicates
    cleaned_df = cleaned_df.drop_duplicates(subset=['user_id'])
    
    # Handle missing values
    cleaned_df['name'] = cleaned_df['name'].fillna('Unknown')
    cleaned_df['review_count'] = cleaned_df['review_count'].fillna(0)
    cleaned_df['average_stars'] = cleaned_df['average_stars'].fillna(cleaned_df['average_stars'].median())
    
    # Extract vote counts if votes column exists
    if 'votes' in cleaned_df.columns:
        cleaned_df['useful_votes'] = cleaned_df['votes'].apply(lambda x: x.get('useful', 0) if isinstance(x, dict) else 0)
        cleaned_df['funny_votes'] = cleaned_df['votes'].apply(lambda x: x.get('funny', 0) if isinstance(x, dict) else 0)
        cleaned_df['cool_votes'] = cleaned_df['votes'].apply(lambda x: x.get('cool', 0) if isinstance(x, dict) else 0)
    
    return cleaned_df

# Apply cleaning functions
print("Cleaning datasets...")
cleaned_reviews = clean_reviews(reviews_df)
cleaned_businesses = clean_businesses(business_df)
cleaned_users = clean_users(users_df)

print("Dataset shapes after cleaning:")
print(f"Reviews: {cleaned_reviews.shape}")
print(f"Businesses: {cleaned_businesses.shape}")
print(f"Users: {cleaned_users.shape}")

# Save cleaned datasets
output_dir = os.path.join(base_path, "cleaned")
os.makedirs(output_dir, exist_ok=True)

cleaned_reviews.to_csv(os.path.join(output_dir, "cleaned_reviews.csv"), index=False)
cleaned_businesses.to_csv(os.path.join(output_dir, "cleaned_businesses.csv"), index=False)
cleaned_users.to_csv(os.path.join(output_dir, "cleaned_users.csv"), index=False)

print("Cleaned datasets saved to:", output_dir)
print("Data cleaning completed.")