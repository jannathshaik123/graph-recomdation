import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define paths to the cleaned datasets
base_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data")
cleaned_dir = os.path.join(base_path, "cleaned")
reviews_path = os.path.join(cleaned_dir, "cleaned_reviews.csv")
business_path = os.path.join(cleaned_dir, "cleaned_businesses.csv")
user_path = os.path.join(cleaned_dir, "cleaned_users.csv")

# Load cleaned datasets
print("Loading cleaned datasets...")
reviews_df = pd.read_csv(reviews_path)
business_df = pd.read_csv(business_path)
users_df = pd.read_csv(user_path)

# Convert date to datetime
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Define preprocessing functions
class DataPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.scaler = StandardScaler()
        
    def preprocess_text(self, text):
        """Preprocess text data"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        
        return ' '.join(tokens)
    
    def extract_text_features(self, texts, max_features=1000, n_components=100):
        """Extract features from text using TF-IDF and dimensionality reduction"""
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=max_features, min_df=5)
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Apply dimensionality reduction
        svd = TruncatedSVD(n_components=n_components)
        text_features = svd.fit_transform(tfidf_matrix)
        
        return text_features, vectorizer, svd
    
    def normalize_numerical_features(self, df, columns):
        """Normalize numerical features"""
        df_copy = df.copy()
        df_copy[columns] = self.scaler.fit_transform(df_copy[columns])
        return df_copy
    
    def encode_categorical_features(self, df, columns):
        """One-hot encode categorical features"""
        df_copy = df.copy()
        
        for col in columns:
            # Get dummies and add prefix
            dummies = pd.get_dummies(df_copy[col], prefix=col, drop_first=True)
            
            # Concatenate with original dataframe
            df_copy = pd.concat([df_copy, dummies], axis=1)
            
            # Drop original column
            df_copy = df_copy.drop(col, axis=1)
            
        return df_copy
    
    def extract_time_features(self, df, date_column):
        """Extract time features from date column"""
        df_copy = df.copy()
        
        df_copy['year'] = df_copy[date_column].dt.year
        df_copy['month'] = df_copy[date_column].dt.month
        df_copy['day'] = df_copy[date_column].dt.day
        df_copy['dayofweek'] = df_copy[date_column].dt.dayofweek
        df_copy['hour'] = df_copy[date_column].dt.hour  # Fixed line
        
        return df_copy

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Preprocess reviews
print("Preprocessing reviews...")
# Extract time features
reviews_df = preprocessor.extract_time_features(reviews_df, 'date')

# Preprocess text
print("Processing review text...")
reviews_df['processed_text'] = reviews_df['text'].apply(preprocessor.preprocess_text)

# Extract text features (limiting to a sample for demonstration)
sample_size = min(10000, len(reviews_df))
sample_reviews = reviews_df.sample(sample_size, random_state=42)
text_features, vectorizer, svd = preprocessor.extract_text_features(sample_reviews['processed_text'])

# Create a dataframe with text features
text_features_df = pd.DataFrame(
    text_features, 
    index=sample_reviews.index, 
    columns=[f'text_feature_{i}' for i in range(text_features.shape[1])]
)

# Normalize numerical features
numerical_cols = ['stars', 'useful_votes', 'funny_votes', 'cool_votes']
reviews_df = preprocessor.normalize_numerical_features(reviews_df, numerical_cols)

# Preprocess businesses
print("Preprocessing businesses...")
# Normalize numerical features
business_numerical_cols = ['stars', 'review_count', 'latitude', 'longitude']
business_df = preprocessor.normalize_numerical_features(
    business_df, 
    [col for col in business_numerical_cols if col in business_df.columns]
)

# Handle categorical features
if 'city' in business_df.columns and 'state' in business_df.columns:
    categorical_cols = ['city', 'state']
    # Limit to top N categories to prevent too many columns
    for col in categorical_cols:
        top_categories = business_df[col].value_counts().head(10).index
        business_df[col] = business_df[col].apply(lambda x: x if x in top_categories else 'Other')
    
    business_df = preprocessor.encode_categorical_features(business_df, categorical_cols)

# Preprocess users
print("Preprocessing users...")
# Normalize numerical features
user_numerical_cols = ['review_count', 'average_stars']
if 'useful_votes' in users_df.columns:
    user_numerical_cols.extend(['useful_votes', 'funny_votes', 'cool_votes'])

users_df = preprocessor.normalize_numerical_features(
    users_df, 
    [col for col in user_numerical_cols if col in users_df.columns]
)

# Create merged dataset for analysis
print("Creating merged dataset...")
# Merge reviews with business data
merged_df = reviews_df.merge(business_df[['business_id', 'name', 'city', 'state', 'stars']], 
                            on='business_id', 
                            suffixes=('_review', '_business'))

# Merge with user data
if 'name' in users_df.columns:
    merged_df = merged_df.merge(users_df[['user_id', 'name', 'review_count', 'average_stars']], 
                               on='user_id', 
                               suffixes=('', '_user'))

# Save preprocessed datasets
output_dir = os.path.join(base_path, "preprocessed")
os.makedirs(output_dir, exist_ok=True)

reviews_df.to_csv(os.path.join(output_dir, "preprocessed_reviews.csv"), index=False)
business_df.to_csv(os.path.join(output_dir, "preprocessed_businesses.csv"), index=False)
users_df.to_csv(os.path.join(output_dir, "preprocessed_users.csv"), index=False)
merged_df.to_csv(os.path.join(output_dir, "preprocessed_merged.csv"), index=False)

# Save text features separately (they can be large)
text_features_df.to_csv(os.path.join(output_dir, "text_features.csv"))

print("Preprocessed datasets saved to:", output_dir)
print("Preprocessing complete!")
