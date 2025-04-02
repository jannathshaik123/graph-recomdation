import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class YelpDataProcessor:
    def __init__(self):
        """Initialize the data processor with reusable components"""
        # Initialize NLTK tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize scalers as instance variables to reuse for test data
        self.review_scaler = StandardScaler()
        self.business_scaler = StandardScaler()
        self.user_scaler = StandardScaler()
        
        # Store column information for consistent processing
        self.review_numerical_cols = ['useful_votes', 'funny_votes', 'cool_votes', 'total_votes', 'text_length']
        self.business_numerical_cols = ['stars', 'review_count', 'latitude', 'longitude']
        self.user_numerical_cols = ['review_count', 'average_stars']
        
        # Store fitted scalers state
        self.is_fitted = False
    
    def load_data(self, reviews_path, business_path, user_path):
        """Load the Yelp dataset from JSON files"""
        print("Loading data...")
        
        reviews_df = self.load_json_data(reviews_path)
        business_df = self.load_json_data(business_path)
        users_df = self.load_json_data(user_path)
        
        print(f"Reviews dataset shape: {reviews_df.shape}")
        print(f"Business dataset shape: {business_df.shape}")
        print(f"Users dataset shape: {users_df.shape}")
        
        return reviews_df, business_df, users_df
    
    def load_json_data(self, file_path):
        """Load JSON data from file"""
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue
        return pd.DataFrame(data)
    
    def preprocess_text(self, text):
        """Preprocess text using NLTK"""
        if not isinstance(text, str):
            return ""
        
        # Lowercase the text
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in self.stop_words]
        
        return ' '.join(tokens)
    
    def get_outlier_bounds(self, df, column):
        """Calculate outlier bounds using IQR method"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return lower_bound, upper_bound
    
    def process_reviews(self, reviews_df, fit=True):
        """Process the reviews dataset"""
        print("Processing reviews dataset...")
        
        # Make a copy to avoid modifying the original
        processed_df = reviews_df.copy()
        
        # Extract vote counts if the 'votes' column exists
        if 'votes' in processed_df.columns:
            processed_df['useful_votes'] = processed_df['votes'].apply(lambda x: x.get('useful', 0) if isinstance(x, dict) else 0)
            processed_df['funny_votes'] = processed_df['votes'].apply(lambda x: x.get('funny', 0) if isinstance(x, dict) else 0)
            processed_df['cool_votes'] = processed_df['votes'].apply(lambda x: x.get('cool', 0) if isinstance(x, dict) else 0)
            processed_df['total_votes'] = processed_df['useful_votes'] + processed_df['funny_votes'] + processed_df['cool_votes']
        else:
            # If 'votes' column is missing, set vote counts to 0
            processed_df['useful_votes'] = 0
            processed_df['funny_votes'] = 0
            processed_df['cool_votes'] = 0
            processed_df['total_votes'] = 0
        
        # Convert date to datetime
        processed_df['date'] = pd.to_datetime(processed_df['date'])
        
        # Add text length feature
        processed_df['text_length'] = processed_df['text'].apply(len)
        
        # Process text using NLTK
        processed_df['processed_text'] = processed_df['text'].apply(self.preprocess_text)
        
        # Remove duplicates
        processed_df = processed_df.drop_duplicates(subset=['review_id'])
        
        # Handle missing values
        processed_df['text'] = processed_df['text'].fillna('')
        processed_df['stars'] = processed_df['stars'].fillna(processed_df['stars'].median())
        
        # Remove reviews with invalid business_id or user_id
        processed_df = processed_df.dropna(subset=['business_id', 'user_id'])
        
        # Handle outliers
        for col in ['text_length', 'total_votes', 'useful_votes', 'funny_votes', 'cool_votes']:
            if fit:
                lower_bound, upper_bound = self.get_outlier_bounds(processed_df, col)
                # Store bounds for test data
                setattr(self, f"{col}_lower", lower_bound)
                setattr(self, f"{col}_upper", upper_bound)
            else:
                # Use stored bounds for test data
                lower_bound = getattr(self, f"{col}_lower")
                upper_bound = getattr(self, f"{col}_upper")
            
            processed_df[col] = processed_df[col].clip(lower=lower_bound if col != 'text_length' else 0, upper=upper_bound)
        
        # Extract time features
        processed_df['year'] = processed_df['date'].dt.year
        processed_df['month'] = processed_df['date'].dt.month
        processed_df['day'] = processed_df['date'].dt.day
        processed_df['dayofweek'] = processed_df['date'].dt.dayofweek
        
        # Normalize numerical features
        if fit:
            self.review_scaler.fit(processed_df[self.review_numerical_cols])
        
        processed_df[self.review_numerical_cols] = self.review_scaler.transform(processed_df[self.review_numerical_cols])
        
        return processed_df
    
    def process_businesses(self, business_df, fit=True):
        """Process the businesses dataset"""
        print("Processing businesses dataset...")
        
        # Make a copy to avoid modifying the original
        processed_df = business_df.copy()
        
        # Remove duplicates
        processed_df = processed_df.drop_duplicates(subset=['business_id'])
        
        # Handle missing values
        processed_df['name'] = processed_df['name'].fillna('Unknown')
        processed_df['city'] = processed_df['city'].fillna('Unknown')
        processed_df['state'] = processed_df['state'].fillna('Unknown')
        processed_df['stars'] = processed_df['stars'].fillna(processed_df['stars'].median())
        processed_df['review_count'] = processed_df['review_count'].fillna(0)
        processed_df['categories'] = processed_df['categories'].fillna('')
        
        # Handle outliers for review_count
        if fit:
            lower_bound, upper_bound = self.get_outlier_bounds(processed_df, 'review_count')
            setattr(self, "review_count_lower", lower_bound)
            setattr(self, "review_count_upper", upper_bound)
        else:
            lower_bound = getattr(self, "review_count_lower")
            upper_bound = getattr(self, "review_count_upper")
        
        processed_df['review_count'] = processed_df['review_count'].clip(lower=lower_bound, upper=upper_bound)
        
        # Convert categories to list
        processed_df['categories_list'] = processed_df['categories'].apply(
            lambda x: [cat.strip() for cat in x.split(',')] if isinstance(x, str) and x else []
        )
        
        # Normalize numerical features
        numerical_cols = [col for col in self.business_numerical_cols if col in processed_df.columns]
        if fit:
            self.business_scaler.fit(processed_df[numerical_cols])
        
        processed_df[numerical_cols] = self.business_scaler.transform(processed_df[numerical_cols])
        
        # Process categorical features
        if fit:
            # Store top cities and states for test data
            self.top_cities = processed_df['city'].value_counts().head(10).index.tolist()
            self.top_categories = ['Restaurants', 'Shopping', 'Food', 'Beauty & Spas', 'Nightlife']
        
        # Create one-hot encoding for top cities
        processed_df['city_processed'] = processed_df['city'].apply(
            lambda x: x if x in self.top_cities else 'Other'
        )
        
        # Extract top categories
        for category in self.top_categories:
            processed_df[f'category_{category}'] = processed_df['categories_list'].apply(
                lambda x: 1 if category in x else 0
            )
        
        return processed_df
    
    def process_users(self, users_df, fit=True):
        """Process the users dataset"""
        print("Processing users dataset...")
        
        # Make a copy to avoid modifying the original
        processed_df = users_df.copy()
        
        # Remove duplicates
        processed_df = processed_df.drop_duplicates(subset=['user_id'])
        
        # Handle missing values
        processed_df['name'] = processed_df['name'].fillna('Unknown')
        processed_df['review_count'] = processed_df['review_count'].fillna(0)
        processed_df['average_stars'] = processed_df['average_stars'].fillna(processed_df['average_stars'].median())
        
        # Extract vote counts if votes column exists
        if 'votes' in processed_df.columns:
            processed_df['useful_votes'] = processed_df['votes'].apply(lambda x: x.get('useful', 0) if isinstance(x, dict) else 0)
            processed_df['funny_votes'] = processed_df['votes'].apply(lambda x: x.get('funny', 0) if isinstance(x, dict) else 0)
            processed_df['cool_votes'] = processed_df['votes'].apply(lambda x: x.get('cool', 0) if isinstance(x, dict) else 0)
            processed_df['total_votes'] = processed_df['useful_votes'] + processed_df['funny_votes'] + processed_df['cool_votes']
            self.user_numerical_cols.extend(['useful_votes', 'funny_votes', 'cool_votes', 'total_votes'])
        
        # Handle outliers
        for col in ['review_count', 'average_stars']:
            if fit:
                lower_bound, upper_bound = self.get_outlier_bounds(processed_df, col)
                setattr(self, f"user_{col}_lower", lower_bound)
                setattr(self, f"user_{col}_upper", upper_bound)
            else:
                lower_bound = getattr(self, f"user_{col}_lower")
                upper_bound = getattr(self, f"user_{col}_upper")
            
            processed_df[col] = processed_df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Normalize numerical features
        numerical_cols = [col for col in self.user_numerical_cols if col in processed_df.columns]
        if fit:
            self.user_scaler.fit(processed_df[numerical_cols])
        
        processed_df[numerical_cols] = self.user_scaler.transform(processed_df[numerical_cols])
        
        return processed_df
    
    def create_merged_dataset(self, reviews_df, business_df, users_df):
        """Create a merged dataset for analysis"""
        # Merge reviews with business data
        merged_df = reviews_df.merge(
            business_df[['business_id', 'name', 'city', 'state', 'stars', 'categories']],
            on='business_id',
            suffixes=('_review', '_business')
        )
        
        # Merge with user data
        merged_df = merged_df.merge(
            users_df[['user_id', 'name', 'review_count', 'average_stars']],
            on='user_id',
            suffixes=('', '_user')
        )
        
        return merged_df
    
    def process_data(self, reviews_path, business_path, user_path, output_dir=None, fit=True):
        """Process all datasets and optionally save them"""
        # Load data
        reviews_df, business_df, users_df = self.load_data(reviews_path, business_path, user_path)
        
        # Process data
        processed_reviews = self.process_reviews(reviews_df, fit)
        processed_businesses = self.process_businesses(business_df, fit)
        processed_users = self.process_users(users_df, fit)
        
        # Create merged dataset
        merged_df = self.create_merged_dataset(processed_reviews, processed_businesses, processed_users)
        
        # Save processed data if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            processed_reviews.to_csv(os.path.join(output_dir, 'processed_reviews.csv'), index=False)
            processed_businesses.to_csv(os.path.join(output_dir, 'processed_businesses.csv'), index=False)
            processed_users.to_csv(os.path.join(output_dir, 'processed_users.csv'), index=False)
            merged_df.to_csv(os.path.join(output_dir, 'merged_dataset.csv'), index=False)
            print(f"Processed data saved to {output_dir}")
        
        # Mark as fitted if this is training data
        if fit:
            self.is_fitted = True
        
        return processed_reviews, processed_businesses, processed_users, merged_df
    
    def process_test_data(self, reviews_path, business_path, user_path, output_dir=None):
        """Process test data using parameters learned from training data"""
        if not self.is_fitted:
            raise ValueError("Processor must be fitted on training data first. Call process_data with fit=True.")
        
        return self.process_data(reviews_path, business_path, user_path, output_dir, fit=False)

# Example usage:
processor = YelpDataProcessor()

# Define file paths
base_path = os.path.join(os.path.dirname(os.getcwd()), "data")
train_reviews_path = os.path.join(base_path, "yelp_training_set/yelp_training_set_review.json")
train_business_path = os.path.join(base_path, "yelp_training_set/yelp_training_set_business.json")
train_user_path = os.path.join(base_path, "yelp_training_set/yelp_training_set_user.json")

test_reviews_path = os.path.join(base_path, "yelp_test_set/yelp_test_set_review.json")
test_business_path = os.path.join(base_path, "yelp_test_set/yelp_test_set_business.json")
test_user_path = os.path.join(base_path, "yelp_test_set/yelp_test_set_user.json")

output_dir = os.path.join(base_path, "cleaned_data")

# Process training data
train_reviews, train_businesses, train_users, train_merged = processor.process_data(
    train_reviews_path, train_business_path, train_user_path, 'processed_train'
)

# Process test data using parameters learned from training data
test_reviews, test_businesses, test_users, test_merged = processor.process_test_data(
    test_reviews_path, test_business_path, test_user_path, 'processed_test'
)
