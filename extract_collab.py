import os
import pandas as pd
import json

# Function to load JSON files line by line (for large files)
def load_json_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def extract_business_data(train_path, test_path, output_path=r'extract\business_data.csv'):
    """Extract business data from Yelp dataset files."""
    print("Extracting business data...")
    
    # Load training business data
    train_business_path = os.path.join(train_path, 'yelp_training_set_business.json')
    if os.path.exists(train_business_path):
        train_business = load_json_file(train_business_path)
    else:
        print(f"Warning: Training business file not found at {train_business_path}")
        train_business = pd.DataFrame()
    
    # Load test business data if it exists
    test_business_path = os.path.join(test_path, 'yelp_test_set_business.json')
    if os.path.exists(test_business_path):
        test_business = load_json_file(test_business_path)
        # Merge training and test data
        if not train_business.empty and not test_business.empty:
            all_business = pd.concat([train_business, test_business], ignore_index=True)
        elif not test_business.empty:
            all_business = test_business
        else:
            all_business = train_business
    else:
        all_business = train_business
    
    if not all_business.empty:
        # Save essential business information
        business_data = all_business[['business_id', 'name', 'full_address', 'city', 'state', 'stars', 'categories']]
        
        # Convert categories to string representation
        business_data['categories'] = business_data['categories'].apply(lambda x: str(x) if x is not None else "[]")
        
        # Save to CSV
        business_data.to_csv(output_path, index=False)
        print(f"Business data saved to {output_path}")
    else:
        print("No business data found.")

def extract_user_data(train_path, test_path, output_path=r'extract\user_data.csv'):
    """Extract user data from Yelp dataset files."""
    print("Extracting user data...")
    
    # Load training user data
    train_user_path = os.path.join(train_path, 'yelp_training_set_user.json')
    if os.path.exists(train_user_path):
        train_user = load_json_file(train_user_path)
    else:
        print(f"Warning: Training user file not found at {train_user_path}")
        train_user = pd.DataFrame()
    
    # Load test user data if it exists
    test_user_path = os.path.join(test_path, 'yelp_test_set_user.json')
    if os.path.exists(test_user_path):
        test_user = load_json_file(test_user_path)
        # Merge training and test data
        if not train_user.empty and not test_user.empty:
            all_user = pd.concat([train_user, test_user], ignore_index=True)
        elif not test_user.empty:
            all_user = test_user
        else:
            all_user = train_user
    else:
        all_user = train_user
    
    if not all_user.empty:
        # Save essential user information
        user_data = all_user[['user_id', 'name', 'review_count', 'average_stars']]
        
        # Fill missing values
        user_data['average_stars'] = user_data['average_stars'].fillna(0)
        
        # Save to CSV
        user_data.to_csv(output_path, index=False)
        print(f"User data saved to {output_path}")
    else:
        print("No user data found.")

if __name__ == "__main__":
    # Set paths to your data
    train_path = os.path.join(os.path.dirname(os.getcwd()),"data/yelp_training_set")  # Replace with your paths
    test_path = os.path.join(os.path.dirname(os.getcwd()),"data/yelp_test_set")        # Replace with your paths
    
    extract_business_data(train_path, test_path)
    extract_user_data(train_path, test_path)