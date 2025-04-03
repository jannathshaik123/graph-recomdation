import json
import pandas as pd
import numpy as np
import os
from datetime import datetime
import pickle

# Define paths - replace these with your actual paths
base_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data")
TRAIN_PATH = os.path.join(os.path.dirname(os.getcwd()), "data/yelp_training_set")
TEST_PATH = os.path.join(os.path.dirname(os.getcwd()), "data/yelp_test_set")
OUTPUT_PATH = os.path.join(os.path.dirname(os.getcwd()),"data/preprocessed_data")

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

def load_json_file(file_path):
    """Load a JSON file with one object per line into a pandas DataFrame."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def process_business_data(train_path, test_path):
    """Process business data from both training and test sets."""
    print("Processing business data...")
    
    # Load training business data
    train_business = load_json_file(os.path.join(train_path, 'yelp_training_set_business.json'))
    
    # Load test business data if it exists
    test_business_path = os.path.join(test_path, 'yelp_test_set_business.json')
    if os.path.exists(test_business_path):
        test_business = load_json_file(test_business_path)
        # Merge training and test data
        test_business['in_test'] = True
        train_business['in_test'] = False
        all_business = pd.concat([train_business, test_business], ignore_index=True)
    else:
        all_business = train_business
        all_business['in_test'] = False
    
    # Process business categories - one-hot encode
    # Extract all unique categories
    all_categories = set()
    for cats in all_business['categories'].dropna():
        all_categories.update(cats)
    
    # Create category features
    for category in all_categories:
        all_business[f'category_{category.replace(" ", "_")}'] = all_business['categories'].apply(
            lambda x: 1 if x is not None and category in x else 0
        )
    
    # Convert neighborhoods list to string
    all_business['neighborhoods'] = all_business['neighborhoods'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) and len(x) > 0 else 'Unknown'
    )
    
    # Fill missing values
    all_business['stars'] = all_business['stars'].fillna(0)  # Missing in test data
    all_business['open'] = all_business['open'].fillna(True)
    
    return all_business

def process_user_data(train_path, test_path):
    """Process user data from both training and test sets."""
    print("Processing user data...")
    
    # Load training user data
    train_user = load_json_file(os.path.join(train_path, 'yelp_training_set_user.json'))
    
    # Load test user data if it exists
    test_user_path = os.path.join(test_path, 'yelp_test_set_user.json')
    if os.path.exists(test_user_path):
        test_user = load_json_file(test_user_path)
        # Test users may be missing some fields like average_stars and votes
        if 'average_stars' not in test_user.columns:
            test_user['average_stars'] = np.nan
        if 'votes' not in test_user.columns:
            test_user['votes'] = [{} for _ in range(len(test_user))]
        
        test_user['in_test'] = True
        train_user['in_test'] = False
        all_user = pd.concat([train_user, test_user], ignore_index=True)
    else:
        all_user = train_user
        all_user['in_test'] = False
    
    # Extract vote features
    all_user['useful_votes'] = all_user['votes'].apply(lambda x: x.get('useful', 0) if x else 0)
    all_user['funny_votes'] = all_user['votes'].apply(lambda x: x.get('funny', 0) if x else 0)
    all_user['cool_votes'] = all_user['votes'].apply(lambda x: x.get('cool', 0) if x else 0)
    all_user['total_votes'] = all_user['useful_votes'] + all_user['funny_votes'] + all_user['cool_votes']
    
    # Fill missing values
    all_user['average_stars'] = all_user['average_stars'].fillna(0)  # Missing in test data
    
    return all_user

def process_review_data(train_path, test_path):
    """Process review data from both training and test sets."""
    print("Processing review data...")
    
    # Load training review data
    train_review = load_json_file(os.path.join(train_path, 'yelp_training_set_review.json'))
    
    # Process date column
    train_review['date'] = pd.to_datetime(train_review['date'])
    train_review['year'] = train_review['date'].dt.year
    train_review['month'] = train_review['date'].dt.month
    train_review['day'] = train_review['date'].dt.day
    train_review['is_weekend'] = train_review['date'].dt.dayofweek >= 5
    
    # Extract vote features for training data
    train_review['useful_votes'] = train_review['votes'].apply(lambda x: x.get('useful', 0) if x else 0)
    train_review['funny_votes'] = train_review['votes'].apply(lambda x: x.get('funny', 0) if x else 0)
    train_review['cool_votes'] = train_review['votes'].apply(lambda x: x.get('cool', 0) if x else 0)
    train_review['total_votes'] = train_review['useful_votes'] + train_review['funny_votes'] + train_review['cool_votes']
    
    # Add basic text features
    train_review['text_length'] = train_review['text'].apply(len)
    train_review['word_count'] = train_review['text'].apply(lambda x: len(x.split()))
    
    # Flag training data
    train_review['in_train'] = True
    
    # Load test review data if it exists
    test_review_path = os.path.join(test_path, 'yelp_test_set_review.json')
    if os.path.exists(test_review_path):
        test_review = load_json_file(test_review_path)
        
        # In test data, we only have user_id and business_id
        test_review['in_train'] = False
        
        # Add empty columns for features not in test data
        for col in train_review.columns:
            if col not in test_review.columns:
                if col in ['date', 'year', 'month', 'day', 'is_weekend']:
                    test_review[col] = pd.NaT
                elif col == 'text':
                    test_review[col] = ''
                elif col in ['stars']:
                    test_review[col] = np.nan  # This is what we'll predict
                else:
                    test_review[col] = 0
        
        # Combine datasets
        all_review = pd.concat([train_review, test_review], ignore_index=True)
    else:
        all_review = train_review
    
    return all_review

def process_checkin_data(train_path, test_path):
    """Process checkin data from both training and test sets."""
    print("Processing checkin data...")
    
    # Load training checkin data
    train_checkin = load_json_file(os.path.join(train_path, 'yelp_training_set_checkin.json'))
    
    # Load test checkin data if it exists
    test_checkin_path = os.path.join(test_path, 'yelp_test_set_checkin.json')
    if os.path.exists(test_checkin_path):
        test_checkin = load_json_file(test_checkin_path)
        test_checkin['in_test'] = True
        train_checkin['in_test'] = False
        all_checkin = pd.concat([train_checkin, test_checkin], ignore_index=True)
    else:
        all_checkin = train_checkin
        all_checkin['in_test'] = False
    
    # Process checkin_info to create aggregated features
    expanded_checkins = []
    
    for _, row in all_checkin.iterrows():
        business_id = row['business_id']
        in_test = row['in_test']
        checkin_info = row['checkin_info']
        
        total_checkins = sum(checkin_info.values())
        
        # Initialize counts for different time periods
        morning_checkins = 0    # 6-11
        afternoon_checkins = 0  # 12-17
        evening_checkins = 0    # 18-23
        night_checkins = 0      # 0-5
        
        weekday_checkins = 0    # Monday-Friday (0-4)
        weekend_checkins = 0    # Saturday-Sunday (5-6)
        
        # Process each time slot
        for time_slot, count in checkin_info.items():
            hour, day = map(int, time_slot.split('-'))
            
            # Time of day
            if 6 <= hour < 12:
                morning_checkins += count
            elif 12 <= hour < 18:
                afternoon_checkins += count
            elif 18 <= hour < 24:
                evening_checkins += count
            else:  # 0 <= hour < 6
                night_checkins += count
            
            # Day of week
            if day < 5:  # Monday-Friday
                weekday_checkins += count
            else:  # Saturday-Sunday
                weekend_checkins += count
        
        expanded_checkins.append({
            'business_id': business_id,
            'in_test': in_test,
            'total_checkins': total_checkins,
            'morning_checkins': morning_checkins,
            'afternoon_checkins': afternoon_checkins,
            'evening_checkins': evening_checkins,
            'night_checkins': night_checkins,
            'weekday_checkins': weekday_checkins,
            'weekend_checkins': weekend_checkins
        })
    
    expanded_checkin_df = pd.DataFrame(expanded_checkins)
    return expanded_checkin_df

def create_user_features(review_df, user_df):
    """Create additional user features based on their review history."""
    print("Creating additional user features...")
    
    # Only use training reviews for this
    train_reviews = review_df[review_df['in_train'] == True]
    
    # Calculate review stats per user
    user_review_stats = train_reviews.groupby('user_id').agg({
        'stars': ['mean', 'std', 'count'],
        'text_length': ['mean', 'max'],
        'word_count': ['mean'],
        'useful_votes': ['sum', 'mean'],
        'funny_votes': ['sum', 'mean'],
        'cool_votes': ['sum', 'mean'],
        'total_votes': ['sum', 'mean']
    })
    
    # Flatten the column names
    user_review_stats.columns = ['_'.join(col).strip() for col in user_review_stats.columns.values]
    user_review_stats = user_review_stats.reset_index()
    
    # Fill NaN values (users with no variance in ratings)
    user_review_stats['stars_std'] = user_review_stats['stars_std'].fillna(0)
    
    # Merge with user_df
    enhanced_user_df = user_df.merge(user_review_stats, on='user_id', how='left')
    
    # Fill NaN values for users in test set with no review history
    for col in user_review_stats.columns:
        if col != 'user_id':
            enhanced_user_df[col] = enhanced_user_df[col].fillna(0)
    
    return enhanced_user_df

def create_business_features(review_df, business_df, checkin_df):
    """Create additional business features based on review history and checkins."""
    print("Creating additional business features...")
    
    # Only use training reviews for this
    train_reviews = review_df[review_df['in_train'] == True]
    
    # Calculate review stats per business
    business_review_stats = train_reviews.groupby('business_id').agg({
        'stars': ['mean', 'std', 'count'],
        'text_length': ['mean'],
        'word_count': ['mean'],
        'useful_votes': ['sum', 'mean'],
        'funny_votes': ['sum', 'mean'],
        'cool_votes': ['sum', 'mean'],
        'total_votes': ['sum', 'mean']
    })
    
    # Flatten the column names
    business_review_stats.columns = ['_'.join(col).strip() for col in business_review_stats.columns.values]
    business_review_stats = business_review_stats.reset_index()
    
    # Fill NaN values (businesses with no variance in ratings)
    business_review_stats['stars_std'] = business_review_stats['stars_std'].fillna(0)
    
    # Merge with business_df
    enhanced_business_df = business_df.merge(business_review_stats, on='business_id', how='left')
    
    # Add checkin features if available
    if checkin_df is not None and not checkin_df.empty:
        checkin_features = checkin_df.drop('in_test', axis=1)
        enhanced_business_df = enhanced_business_df.merge(checkin_features, on='business_id', how='left')
        
        # Fill NaN checkin values
        for col in checkin_features.columns:
            if col != 'business_id':
                enhanced_business_df[col] = enhanced_business_df[col].fillna(0)
    
    # Fill NaN values for businesses in test set with no review history
    for col in business_review_stats.columns:
        if col != 'business_id':
            enhanced_business_df[col] = enhanced_business_df[col].fillna(0)
    
    return enhanced_business_df

def prepare_final_datasets(review_df, enhanced_user_df, enhanced_business_df):
    """Prepare final training and test datasets for the recommendation system."""
    print("Preparing final datasets...")
    
    # Separate train and test reviews
    train_reviews = review_df[review_df['in_train'] == True].copy()
    test_reviews = review_df[review_df['in_train'] == False].copy()
    
    # Create feature dataframes by merging with user and business features
    # Training features
    train_features = train_reviews[['user_id', 'business_id', 'stars']].copy()
    
    # Test features
    test_features = test_reviews[['user_id', 'business_id']].copy()
    
    # Merge user features
    user_features = enhanced_user_df.drop(['name', 'review_count', 'votes', 'in_test'], axis=1, errors='ignore')
    train_features = train_features.merge(user_features, on='user_id', how='left')
    test_features = test_features.merge(user_features, on='user_id', how='left')
    
    # Merge business features
    business_features = enhanced_business_df.drop(['name', 'neighborhoods', 'full_address', 
                                                'city', 'state', 'latitude', 'longitude', 
                                                'review_count', 'categories', 'open', 'in_test'], 
                                               axis=1, errors='ignore')
    train_features = train_features.merge(business_features, on='business_id', how='left')
    test_features = test_features.merge(business_features, on='business_id', how='left')
    
    # Fill any remaining NaN values
    train_features = train_features.fillna(0)
    test_features = test_features.fillna(0)
    
    return train_features, test_features

def main():
    """Main function to run the preprocessing pipeline."""
    start_time = datetime.now()
    print(f"Starting preprocessing at {start_time}")
    
    # Process all data files
    business_df = process_business_data(TRAIN_PATH, TEST_PATH)
    user_df = process_user_data(TRAIN_PATH, TEST_PATH)
    review_df = process_review_data(TRAIN_PATH, TEST_PATH)
    checkin_df = process_checkin_data(TRAIN_PATH, TEST_PATH)
    
    # Create enhanced features
    enhanced_user_df = create_user_features(review_df, user_df)
    enhanced_business_df = create_business_features(review_df, business_df, checkin_df)
    
    # Prepare final datasets
    train_features, test_features = prepare_final_datasets(review_df, enhanced_user_df, enhanced_business_df)
    
    # Save all processed data
    print(f"Saving processed data to {OUTPUT_PATH}...")
    
    # Save full processed dataframes
    business_df.to_csv(os.path.join(OUTPUT_PATH, 'processed_business.csv'), index=False)
    user_df.to_csv(os.path.join(OUTPUT_PATH, 'processed_user.csv'), index=False)
    review_df.to_csv(os.path.join(OUTPUT_PATH, 'processed_review.csv'), index=False)
    checkin_df.to_csv(os.path.join(OUTPUT_PATH, 'processed_checkin.csv'), index=False)
    enhanced_user_df.to_csv(os.path.join(OUTPUT_PATH, 'enhanced_user.csv'), index=False)
    enhanced_business_df.to_csv(os.path.join(OUTPUT_PATH, 'enhanced_business.csv'), index=False)
    
    # Save train and test feature sets
    train_features.to_csv(os.path.join(OUTPUT_PATH, 'train_features.csv'), index=False)
    test_features.to_csv(os.path.join(OUTPUT_PATH, 'test_features.csv'), index=False)
    
    # Also save as pickle for easier loading in ML code
    with open(os.path.join(OUTPUT_PATH, 'train_features.pkl'), 'wb') as f:
        pickle.dump(train_features, f)
    
    with open(os.path.join(OUTPUT_PATH, 'test_features.pkl'), 'wb') as f:
        pickle.dump(test_features, f)
        
    # Prepare a sample submission file
    sample_submission = test_features[['user_id', 'business_id']].copy()
    # Default prediction (mean of training ratings)
    sample_submission['stars'] = train_features['stars'].mean()
    sample_submission.to_csv(os.path.join(OUTPUT_PATH, 'sample_submission.csv'), index=False)
    
    end_time = datetime.now()
    print(f"Preprocessing completed in {end_time - start_time}")
    print(f"All processed data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()