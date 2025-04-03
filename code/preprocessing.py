import json
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Function to read JSON files line by line
def read_json_lines(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Function to check if file exists
def check_file_exists(file_path):
    return os.path.isfile(file_path)

# Define file paths
# You'll need to update these paths to match your directory structure
base_path = os.path.join(os.path.dirname(os.getcwd()), "data")

train_business_path = os.path.join(base_path, "yelp_training_set/yelp_training_set_business.json")
train_user_path = os.path.join(base_path, "yelp_training_set/yelp_training_set_user.json")
train_review_path = os.path.join(base_path, "yelp_training_set/yelp_training_set_review.json")
train_checkin_path = os.path.join(base_path, "yelp_training_set/yelp_training_set_checkin.json")

test_business_path = os.path.join(base_path, "yelp_test_set/yelp_test_set_business.json")
test_user_path = os.path.join(base_path, "yelp_test_set/yelp_test_set_user.json")
test_review_path = os.path.join(base_path, "yelp_test_set/yelp_test_set_review.json")
test_checkin_path = os.path.join(base_path, "yelp_test_set/yelp_test_set_checkin.json")

print("Starting data preprocessing...")

# Load test data
print("Loading test data...")
test_businesses = read_json_lines(test_business_path)
test_users = read_json_lines(test_user_path)
test_reviews = read_json_lines(test_review_path)
test_checkins = read_json_lines(test_checkin_path) if check_file_exists(test_checkin_path) else []

# Load training data
print("Loading training data...")
train_businesses = read_json_lines(train_business_path)
train_users = read_json_lines(train_user_path)
train_reviews = read_json_lines(train_review_path)
train_checkins = read_json_lines(train_checkin_path) if check_file_exists(train_checkin_path) else []

# Convert to DataFrames
print("Converting to DataFrames...")
business_train_df = pd.DataFrame(train_businesses)
user_train_df = pd.DataFrame(train_users)
review_train_df = pd.DataFrame(train_reviews)
checkin_train_df = pd.DataFrame(train_checkins) if train_checkins else pd.DataFrame()

business_test_df = pd.DataFrame(test_businesses)
user_test_df = pd.DataFrame(test_users)
review_test_df = pd.DataFrame(test_reviews)
checkin_test_df = pd.DataFrame(test_checkins) if test_checkins else pd.DataFrame()

# Combine business data
print("Merging business data...")
all_businesses = pd.concat([business_train_df, business_test_df], ignore_index=True)
all_businesses = all_businesses.drop_duplicates(subset='business_id')

# Combine user data
print("Merging user data...")
all_users = pd.concat([user_train_df, user_test_df], ignore_index=True)
all_users = all_users.drop_duplicates(subset='user_id')

# Combine review data
print("Merging review data...")
all_reviews = pd.concat([review_train_df, review_test_df], ignore_index=True)

# Combine checkin data if available
if not checkin_train_df.empty or not checkin_test_df.empty:
    print("Merging checkin data...")
    all_checkins = pd.concat([checkin_train_df, checkin_test_df], ignore_index=True)
    all_checkins = all_checkins.drop_duplicates(subset='business_id')
else:
    all_checkins = pd.DataFrame()

# Process business features
print("Processing business features...")
# Convert categories from list to string if needed
all_businesses['categories_str'] = all_businesses['categories'].apply(
    lambda x: ', '.join(x) if isinstance(x, list) else '')

# Extract business features
business_features = all_businesses[['business_id', 'name', 'city', 'state', 
                                   'latitude', 'longitude', 'review_count', 'categories_str']]

# Add stars column if it exists (it's removed in test data)
if 'stars' in all_businesses.columns:
    business_features['stars'] = all_businesses['stars']

# Process user features
print("Processing user features...")
user_features = all_users[['user_id', 'name', 'review_count']]

# Add average_stars and votes if they exist (they're removed in test data)
if 'average_stars' in user_train_df.columns:
    # Create a mapping of user_id to average_stars from training data
    avg_stars_map = user_train_df.set_index('user_id')['average_stars'].to_dict()
    # Apply this mapping to all users
    user_features['average_stars'] = user_features['user_id'].map(avg_stars_map)

if 'votes' in user_train_df.columns:
    # Extract vote counts from training data
    user_train_votes = user_train_df.copy()
    user_train_votes['useful_votes'] = user_train_votes['votes'].apply(lambda x: x.get('useful', 0) if isinstance(x, dict) else 0)
    user_train_votes['funny_votes'] = user_train_votes['votes'].apply(lambda x: x.get('funny', 0) if isinstance(x, dict) else 0)
    user_train_votes['cool_votes'] = user_train_votes['votes'].apply(lambda x: x.get('cool', 0) if isinstance(x, dict) else 0)
    
    # Create mappings
    useful_map = user_train_votes.set_index('user_id')['useful_votes'].to_dict()
    funny_map = user_train_votes.set_index('user_id')['funny_votes'].to_dict()
    cool_map = user_train_votes.set_index('user_id')['cool_votes'].to_dict()
    
    # Apply mappings
    user_features['useful_votes'] = user_features['user_id'].map(useful_map)
    user_features['funny_votes'] = user_features['user_id'].map(funny_map)
    user_features['cool_votes'] = user_features['user_id'].map(cool_map)

# Process review features
print("Processing review features...")
# For training reviews, keep essential columns
review_features = review_train_df[['user_id', 'business_id', 'stars', 'date']]

# Convert date to datetime and extract temporal features
review_features['date'] = pd.to_datetime(review_features['date'])
review_features['year'] = review_features['date'].dt.year
review_features['month'] = review_features['date'].dt.month
review_features['day_of_week'] = review_features['date'].dt.dayofweek

# Process votes in reviews if available
if 'votes' in review_train_df.columns:
    review_features['useful_votes'] = review_train_df['votes'].apply(lambda x: x.get('useful', 0) if isinstance(x, dict) else 0)
    review_features['funny_votes'] = review_train_df['votes'].apply(lambda x: x.get('funny', 0) if isinstance(x, dict) else 0)
    review_features['cool_votes'] = review_train_df['votes'].apply(lambda x: x.get('cool', 0) if isinstance(x, dict) else 0)

# Create a merged dataset for recommendation
print("Creating merged dataset...")
merged_data = review_features.merge(business_features, on='business_id', how='left')
merged_data = merged_data.merge(user_features, on='user_id', how='left', suffixes=('_business', '_user'))

# Save the merged dataset
print("Saving merged dataset...")
merged_data.to_csv('yelp_recommendation_data.csv', index=False)

# Create a user-item matrix for collaborative filtering
print("Creating user-item matrix...")
user_item_matrix = review_features.pivot_table(
    index='user_id', 
    columns='business_id', 
    values='stars'
)
user_item_matrix.to_csv('user_item_matrix.csv')

# Prepare test set for predictions
print("Preparing test set for predictions...")
prediction_pairs = review_test_df[['user_id', 'business_id']]
prediction_pairs.to_csv('prediction_pairs.csv', index=False)

# Create a business profile dataset
print("Creating business profile dataset...")
business_profile = business_features.copy()
business_profile.to_csv('business_profile.csv', index=False)

# Create a user profile dataset
print("Creating user profile dataset...")
user_profile = user_features.copy()
user_profile.to_csv('user_profile.csv', index=False)

# If checkin data is available, process it
if not all_checkins.empty:
    print("Processing checkin data...")
    # Flatten the checkin_info dictionary into columns
    checkin_features = all_checkins[['business_id']].copy()
    
    # Create a function to extract checkin counts by day and hour
    def extract_checkin_features(checkin_info):
        if not isinstance(checkin_info, dict):
            return {}
        
        # Initialize counts
        day_counts = {i: 0 for i in range(7)}  # 0-6 for days of week
        hour_counts = {i: 0 for i in range(24)}  # 0-23 for hours
        
        for key, count in checkin_info.items():
            if '-' in key:
                hour, day = map(int, key.split('-'))
                day_counts[day] += count
                hour_counts[hour] += count
                
        return {
            **{f'day_{day}': count for day, count in day_counts.items()},
            **{f'hour_{hour}': count for hour, count in hour_counts.items()}
        }
    
    # Apply the function to each row
    checkin_dicts = all_checkins['checkin_info'].apply(extract_checkin_features)
    checkin_df = pd.DataFrame(checkin_dicts.tolist())
    
    # Combine with business_id
    checkin_features = pd.concat([checkin_features, checkin_df], axis=1)
    checkin_features.to_csv('checkin_features.csv', index=False)
    
    # Merge checkin data with business profile
    business_with_checkins = business_profile.merge(checkin_features, on='business_id', how='left')
    business_with_checkins.to_csv('business_with_checkins.csv', index=False)

# Create a combined dataset with all available features
print("Creating final combined dataset...")
final_dataset = merged_data.copy()

# Add any additional features from checkins if available
if not all_checkins.empty:
    # Get total checkin count per business
    checkin_counts = checkin_features.copy()
    day_cols = [col for col in checkin_counts.columns if col.startswith('day_')]
    hour_cols = [col for col in checkin_counts.columns if col.startswith('hour_')]
    
    if day_cols and hour_cols:
        checkin_counts['total_checkins'] = checkin_counts[day_cols + hour_cols].sum(axis=1)
        checkin_counts = checkin_counts[['business_id', 'total_checkins']]
        
        # Merge with final dataset
        final_dataset = final_dataset.merge(checkin_counts, on='business_id', how='left')
        final_dataset['total_checkins'] = final_dataset['total_checkins'].fillna(0)

# Save the final combined dataset
final_dataset.to_csv('yelp_recommendation_final.csv', index=False)

print("Data preprocessing complete! Files saved:")
print("1. yelp_recommendation_data.csv - Main merged dataset")
print("2. user_item_matrix.csv - For collaborative filtering")
print("3. prediction_pairs.csv - Test set user-business pairs for prediction")
print("4. business_profile.csv - Business features")
print("5. user_profile.csv - User features")
if not all_checkins.empty:
    print("6. checkin_features.csv - Checkin features")
    print("7. business_with_checkins.csv - Business data with checkin information")
print("8. yelp_recommendation_final.csv - Final combined dataset with all features")

print("\nThese files can now be used to build various recommendation models!")
