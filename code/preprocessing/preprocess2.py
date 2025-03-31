import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import nltk
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Set plot style
plt.style.use('ggplot')
sns.set_palette('Set2')

class YelpDataCleaner:
    def __init__(self, reviews_path, business_path, user_path, output_dir):
        """Initialize the data cleaner with file paths"""
        self.reviews_path = reviews_path
        self.business_path = business_path
        self.user_path = user_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize DataFrames
        self.reviews_df = None
        self.business_df = None
        self.users_df = None
        
        # Initialize cleaned DataFrames
        self.cleaned_reviews_df = None
        self.cleaned_business_df = None
        self.cleaned_users_df = None
        
        # Initialize NLTK tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def load_data(self):
        """Load the Yelp dataset from JSON files"""
        print("Loading data...")
        
        self.reviews_df = self.load_json_data(self.reviews_path)
        self.business_df = self.load_json_data(self.business_path)
        self.users_df = self.load_json_data(self.user_path)
        
        print(f"Reviews dataset shape: {self.reviews_df.shape}")
        print(f"Business dataset shape: {self.business_df.shape}")
        print(f"Users dataset shape: {self.users_df.shape}")
        
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
    
    def clean_reviews(self):
        """Clean the reviews dataset"""
        print("Cleaning reviews dataset...")
        
        # Make a copy to avoid modifying the original
        self.cleaned_reviews_df = self.reviews_df.copy()
        
        # Extract vote counts
        self.cleaned_reviews_df['useful_votes'] = self.cleaned_reviews_df['votes'].apply(lambda x: x.get('useful', 0))
        self.cleaned_reviews_df['funny_votes'] = self.cleaned_reviews_df['votes'].apply(lambda x: x.get('funny', 0))
        self.cleaned_reviews_df['cool_votes'] = self.cleaned_reviews_df['votes'].apply(lambda x: x.get('cool', 0))
        self.cleaned_reviews_df['total_votes'] = (self.cleaned_reviews_df['useful_votes'] + 
                                                 self.cleaned_reviews_df['funny_votes'] + 
                                                 self.cleaned_reviews_df['cool_votes'])
        
        # Convert date to datetime
        self.cleaned_reviews_df['date'] = pd.to_datetime(self.cleaned_reviews_df['date'])
        
        # Add text length feature
        self.cleaned_reviews_df['text_length'] = self.cleaned_reviews_df['text'].apply(len)
        
        # Process text using NLTK
        self.cleaned_reviews_df['processed_text'] = self.cleaned_reviews_df['text'].apply(self.preprocess_text)
        
        # Remove duplicates
        self.cleaned_reviews_df = self.cleaned_reviews_df.drop_duplicates(subset=['review_id'])
        
        # Handle missing values
        self.cleaned_reviews_df['text'] = self.cleaned_reviews_df['text'].fillna('')
        self.cleaned_reviews_df['stars'] = self.cleaned_reviews_df['stars'].fillna(self.cleaned_reviews_df['stars'].median())
        
        # Remove reviews with invalid business_id or user_id
        self.cleaned_reviews_df = self.cleaned_reviews_df.dropna(subset=['business_id', 'user_id'])
        
        # Detect and handle outliers
        self.handle_review_outliers()
        
        print(f"Cleaned reviews dataset shape: {self.cleaned_reviews_df.shape}")
        
    def clean_businesses(self):
        """Clean the businesses dataset"""
        print("Cleaning businesses dataset...")
        
        # Make a copy to avoid modifying the original
        self.cleaned_business_df = self.business_df.copy()
        
        # Remove duplicates
        self.cleaned_business_df = self.cleaned_business_df.drop_duplicates(subset=['business_id'])
        
        # Handle missing values
        self.cleaned_business_df['name'] = self.cleaned_business_df['name'].fillna('Unknown')
        self.cleaned_business_df['city'] = self.cleaned_business_df['city'].fillna('Unknown')
        self.cleaned_business_df['state'] = self.cleaned_business_df['state'].fillna('Unknown')
        self.cleaned_business_df['stars'] = self.cleaned_business_df['stars'].fillna(self.cleaned_business_df['stars'].median())
        self.cleaned_business_df['review_count'] = self.cleaned_business_df['review_count'].fillna(0)
        
        # Process categories
        self.cleaned_business_df['categories'] = self.cleaned_business_df['categories'].fillna('')
        
        # Convert categories to list if it's a string
        if self.cleaned_business_df['categories'].dtype == object:
            self.cleaned_business_df['categories_list'] = self.cleaned_business_df['categories'].apply(
                lambda x: [cat.strip() for cat in x.split(',')] if isinstance(x, str) and x else []
            )
        
        # Handle open status
        self.cleaned_business_df['open'] = self.cleaned_business_df['open'].fillna(True)
        
        # Detect and handle outliers
        self.handle_business_outliers()
        
        print(f"Cleaned businesses dataset shape: {self.cleaned_business_df.shape}")
        
    def clean_users(self):
        """Clean the users dataset"""
        print("Cleaning users dataset...")
        
        # Make a copy to avoid modifying the original
        self.cleaned_users_df = self.users_df.copy()
        
        # Remove duplicates
        self.cleaned_users_df = self.cleaned_users_df.drop_duplicates(subset=['user_id'])
        
        # Handle missing values
        self.cleaned_users_df['name'] = self.cleaned_users_df['name'].fillna('Unknown')
        self.cleaned_users_df['review_count'] = self.cleaned_users_df['review_count'].fillna(0)
        self.cleaned_users_df['average_stars'] = self.cleaned_users_df['average_stars'].fillna(self.cleaned_users_df['average_stars'].median())
        
        # Extract vote counts if votes column exists
        if 'votes' in self.cleaned_users_df.columns:
            self.cleaned_users_df['useful_votes'] = self.cleaned_users_df['votes'].apply(lambda x: x.get('useful', 0))
            self.cleaned_users_df['funny_votes'] = self.cleaned_users_df['votes'].apply(lambda x: x.get('funny', 0))
            self.cleaned_users_df['cool_votes'] = self.cleaned_users_df['votes'].apply(lambda x: x.get('cool', 0))
            self.cleaned_users_df['total_votes'] = (self.cleaned_users_df['useful_votes'] + 
                                                   self.cleaned_users_df['funny_votes'] + 
                                                   self.cleaned_users_df['cool_votes'])
        
        # Detect and handle outliers
        self.handle_user_outliers()
        
        print(f"Cleaned users dataset shape: {self.cleaned_users_df.shape}")
    
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
    
    def handle_review_outliers(self):
        """Handle outliers in the reviews dataset"""
        # Handle text length outliers
        lower_bound, upper_bound = self.get_outlier_bounds(self.cleaned_reviews_df, 'text_length')
        text_length_outliers = self.cleaned_reviews_df[(self.cleaned_reviews_df['text_length'] < lower_bound) | 
                                                      (self.cleaned_reviews_df['text_length'] > upper_bound)]
        print(f"Text length outliers: {len(text_length_outliers)} ({len(text_length_outliers)/len(self.cleaned_reviews_df)*100:.2f}%)")
        
        # Handle vote outliers
        lower_bound, upper_bound = self.get_outlier_bounds(self.cleaned_reviews_df, 'total_votes')
        vote_outliers = self.cleaned_reviews_df[(self.cleaned_reviews_df['total_votes'] < lower_bound) | 
                                               (self.cleaned_reviews_df['total_votes'] > upper_bound)]
        print(f"Vote outliers: {len(vote_outliers)} ({len(vote_outliers)/len(self.cleaned_reviews_df)*100:.2f}%)")
        
        # Instead of removing outliers, cap them at the bounds
        self.cleaned_reviews_df['text_length'] = self.cleaned_reviews_df['text_length'].clip(lower=lower_bound, upper=upper_bound)
        
        lower_bound, upper_bound = self.get_outlier_bounds(self.cleaned_reviews_df, 'total_votes')
        self.cleaned_reviews_df['total_votes'] = self.cleaned_reviews_df['total_votes'].clip(lower=lower_bound, upper=upper_bound)
        self.cleaned_reviews_df['useful_votes'] = self.cleaned_reviews_df['useful_votes'].clip(lower=0, upper=upper_bound)
        self.cleaned_reviews_df['funny_votes'] = self.cleaned_reviews_df['funny_votes'].clip(lower=0, upper=upper_bound)
        self.cleaned_reviews_df['cool_votes'] = self.cleaned_reviews_df['cool_votes'].clip(lower=0, upper=upper_bound)
    
    def handle_business_outliers(self):
        """Handle outliers in the businesses dataset"""
        # Handle review count outliers
        lower_bound, upper_bound = self.get_outlier_bounds(self.cleaned_business_df, 'review_count')
        review_count_outliers = self.cleaned_business_df[(self.cleaned_business_df['review_count'] < lower_bound) | 
                                                        (self.cleaned_business_df['review_count'] > upper_bound)]
        print(f"Business review count outliers: {len(review_count_outliers)} ({len(review_count_outliers)/len(self.cleaned_business_df)*100:.2f}%)")
        
        # Cap review counts at the bounds
        self.cleaned_business_df['review_count'] = self.cleaned_business_df['review_count'].clip(lower=lower_bound, upper=upper_bound)
    
    def handle_user_outliers(self):
        """Handle outliers in the users dataset"""
        # Handle review count outliers
        lower_bound, upper_bound = self.get_outlier_bounds(self.cleaned_users_df, 'review_count')
        review_count_outliers = self.cleaned_users_df[(self.cleaned_users_df['review_count'] < lower_bound) | 
                                                     (self.cleaned_users_df['review_count'] > upper_bound)]
        print(f"User review count outliers: {len(review_count_outliers)} ({len(review_count_outliers)/len(self.cleaned_users_df)*100:.2f}%)")
        
        # Cap review counts at the bounds
        self.cleaned_users_df['review_count'] = self.cleaned_users_df['review_count'].clip(lower=lower_bound, upper=upper_bound)
        
        # Handle average stars outliers
        lower_bound, upper_bound = self.get_outlier_bounds(self.cleaned_users_df, 'average_stars')
        stars_outliers = self.cleaned_users_df[(self.cleaned_users_df['average_stars'] < lower_bound) | 
                                              (self.cleaned_users_df['average_stars'] > upper_bound)]
        print(f"User average stars outliers: {len(stars_outliers)} ({len(stars_outliers)/len(self.cleaned_users_df)*100:.2f}%)")
        
        # Cap average stars at the bounds
        self.cleaned_users_df['average_stars'] = self.cleaned_users_df['average_stars'].clip(lower=lower_bound, upper=upper_bound)
    
    def preprocess_data(self):
        """Preprocess the cleaned data"""
        print("Preprocessing data...")
        
        # Preprocess reviews
        self.preprocess_reviews()
        
        # Preprocess businesses
        self.preprocess_businesses()
        
        # Preprocess users
        self.preprocess_users()
    
    def preprocess_reviews(self):
        """Preprocess the reviews dataset"""
        # Extract time features
        self.cleaned_reviews_df['year'] = self.cleaned_reviews_df['date'].dt.year
        self.cleaned_reviews_df['month'] = self.cleaned_reviews_df['date'].dt.month
        self.cleaned_reviews_df['day'] = self.cleaned_reviews_df['date'].dt.day
        self.cleaned_reviews_df['dayofweek'] = self.cleaned_reviews_df['date'].dt.dayofweek
        
        # Normalize numerical features
        scaler = StandardScaler()
        numerical_cols = ['useful_votes', 'funny_votes', 'cool_votes', 'total_votes', 'text_length']
        self.cleaned_reviews_df[numerical_cols] = scaler.fit_transform(self.cleaned_reviews_df[numerical_cols])
    
    def preprocess_businesses(self):
        """Preprocess the businesses dataset"""
        # Normalize numerical features
        scaler = StandardScaler()
        numerical_cols = ['stars', 'review_count', 'latitude', 'longitude']
        self.cleaned_business_df[numerical_cols] = scaler.fit_transform(self.cleaned_business_df[numerical_cols])
        
        # Create one-hot encoding for top cities
        top_cities = self.cleaned_business_df['city'].value_counts().head(10).index
        self.cleaned_business_df['city_processed'] = self.cleaned_business_df['city'].apply(
            lambda x: x if x in top_cities else 'Other'
        )
        city_dummies = pd.get_dummies(self.cleaned_business_df['city_processed'], prefix='city')
        self.cleaned_business_df = pd.concat([self.cleaned_business_df, city_dummies], axis=1)
        
        # Create one-hot encoding for top states
        state_dummies = pd.get_dummies(self.cleaned_business_df['state'], prefix='state')
        self.cleaned_business_df = pd.concat([self.cleaned_business_df, state_dummies], axis=1)
        
        # Extract top categories
        if 'categories_list' in self.cleaned_business_df.columns:
            top_categories = ['Restaurants', 'Shopping', 'Food', 'Beauty & Spas', 'Nightlife']
            
            for category in top_categories:
                self.cleaned_business_df[f'category_{category}'] = self.cleaned_business_df['categories_list'].apply(
                    lambda x: 1 if category in x else 0
                )
    
    def preprocess_users(self):
        """Preprocess the users dataset"""
        # Normalize numerical features
        scaler = StandardScaler()
        numerical_cols = ['review_count', 'average_stars']
        
        if 'useful_votes' in self.cleaned_users_df.columns:
            numerical_cols.extend(['useful_votes', 'funny_votes', 'cool_votes', 'total_votes'])
            
        self.cleaned_users_df[numerical_cols] = scaler.fit_transform(self.cleaned_users_df[numerical_cols])
    
    def save_data(self):
        """Save the cleaned and preprocessed data"""
        print("Saving cleaned and preprocessed data...")
        
        # Save reviews
        self.cleaned_reviews_df.to_csv(os.path.join(self.output_dir, 'cleaned_reviews.csv'), index=False)
        
        # Save businesses
        self.cleaned_business_df.to_csv(os.path.join(self.output_dir, 'cleaned_businesses.csv'), index=False)
        
        # Save users
        self.cleaned_users_df.to_csv(os.path.join(self.output_dir, 'cleaned_users.csv'), index=False)
        
        # Create and save a merged dataset
        self.create_and_save_merged_dataset()
        
        print(f"Data saved to {self.output_dir}")
    
    def create_and_save_merged_dataset(self):
        """Create and save a merged dataset for analysis"""
        # Merge reviews with business data
        merged_df = self.cleaned_reviews_df.merge(
            self.cleaned_business_df[['business_id', 'name', 'city', 'state', 'stars', 'categories']],
            on='business_id',
            suffixes=('_review', '_business')
        )
        
        # Merge with user data
        merged_df = merged_df.merge(
            self.cleaned_users_df[['user_id', 'name', 'review_count', 'average_stars']],
            on='user_id',
            suffixes=('', '_user')
        )
        
        # Save merged dataset
        merged_df.to_csv(os.path.join(self.output_dir, 'merged_dataset.csv'), index=False)
    
    def run_pipeline(self):
        """Run the complete data cleaning and preprocessing pipeline"""
        # Load data
        self.load_data()
        
        # Clean data
        self.clean_reviews()
        self.clean_businesses()
        self.clean_users()
        
        # Preprocess data
        self.preprocess_data()
        
        # Save data
        self.save_data()
        
        print("Data cleaning and preprocessing pipeline completed!")

if __name__ == "__main__":
    def generate_summary_statistics(cleaner):
        """Generate summary statistics for the cleaned data"""
        print("\n=== SUMMARY STATISTICS ===\n")

        # Reviews summary
        print("Reviews Dataset:")
        print(f"Total reviews: {len(cleaner.cleaned_reviews_df)}")
        print(f"Average rating: {cleaner.cleaned_reviews_df['stars'].mean():.2f}")
        print(f"Rating distribution: {cleaner.cleaned_reviews_df['stars'].value_counts().sort_index().to_dict()}")
        print(f"Average votes per review: {cleaner.cleaned_reviews_df['total_votes'].mean():.2f}")
        print(f"Average text length: {cleaner.cleaned_reviews_df['text_length'].mean():.2f}")

        # Business summary
        print("\nBusiness Dataset:")
        print(f"Total businesses: {len(cleaner.cleaned_business_df)}")
        print(f"Average rating: {cleaner.cleaned_business_df['stars'].mean():.2f}")
        print(f"Average review count: {cleaner.cleaned_business_df['review_count'].mean():.2f}")
        print(f"Top 5 cities: {cleaner.cleaned_business_df['city'].value_counts().head(5).to_dict()}")
        print(f"Top 5 states: {cleaner.cleaned_business_df['state'].value_counts().head(5).to_dict()}")

        # Users summary
        print("\nUsers Dataset:")
        print(f"Total users: {len(cleaner.cleaned_users_df)}")
        print(f"Average reviews per user: {cleaner.cleaned_users_df['review_count'].mean():.2f}")
        print(f"Average user rating: {cleaner.cleaned_users_df['average_stars'].mean():.2f}")
        
        
    # Define file paths
    base_path = os.path.join(os.path.dirname(os.getcwd()), "data")
    reviews_path = os.path.join(base_path, "yelp_training_set/yelp_training_set_review.json")
    business_path = os.path.join(base_path, "yelp_training_set/yelp_training_set_business.json")
    user_path = os.path.join(base_path, "yelp_training_set/yelp_training_set_user.json")
    output_dir = os.path.join(base_path, "cleaned_data")

    # Create and run the data cleaner
    cleaner = YelpDataCleaner(reviews_path, business_path, user_path, output_dir)
    cleaner.run_pipeline()

    # Generate summary statistics
    generate_summary_statistics(cleaner)