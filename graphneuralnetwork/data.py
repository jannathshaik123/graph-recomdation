import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from neo4j import GraphDatabase
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

class Neo4jDataLoader:
    """
    Loads data from Neo4j graph database and converts it to PyTorch Geometric format.
    Implements memory-efficient loading through batching and streaming.
    """
    def __init__(self, uri, user, password, batch_size=1000):
        """
        Initialize the data loader.
        
        Args:
            uri: Neo4j database URI
            user: Neo4j username
            password: Neo4j password
            batch_size: Batch size for loading data from Neo4j
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.batch_size = batch_size
        self.encoders = {}
        self.scalers = {}
        self.node_mapping = {}  
        self.reverse_mapping = {}  
        self.node_types = {}  
        
    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()
        
    def load_graph_data(self, include_features=True, cache_dir=None):
        """
        Load the graph data from Neo4j.
        
        Args:
            include_features: Whether to include node features
            cache_dir: Directory to cache loaded data for faster loading
            
        Returns:
            PyTorch Geometric Data object
        """
        
        if cache_dir and os.path.exists(os.path.join(cache_dir, 'graph_data.pt')):
            print("Loading cached graph data...")
            return torch.load(os.path.join(cache_dir, 'graph_data.pt'))
        
        print("Loading graph data from Neo4j...")
        
        
        user_features, business_features = None, None
        if include_features:
            user_features, user_mapping = self._load_users()
            business_features, business_mapping = self._load_businesses()
            
            
            user_offset = 0
            business_offset = len(user_mapping)
            
            
            for user_id, idx in user_mapping.items():
                self.node_mapping[user_id] = idx
                self.reverse_mapping[idx] = user_id
                self.node_types[idx] = 'user'
                
            for business_id, idx in business_mapping.items():
                self.node_mapping[business_id] = idx + business_offset
                self.reverse_mapping[idx + business_offset] = business_id
                self.node_types[idx + business_offset] = 'business'
        
        
        edge_index, edge_attr = self._load_reviews()
        
        
        if include_features:
            num_nodes = len(self.node_mapping)
            feature_dim = max(user_features.shape[1], business_features.shape[1])
            
            
            if user_features.shape[1] < feature_dim:
                padding = np.zeros((user_features.shape[0], feature_dim - user_features.shape[1]))
                user_features = np.hstack([user_features, padding])
            if business_features.shape[1] < feature_dim:
                padding = np.zeros((business_features.shape[0], feature_dim - business_features.shape[1]))
                business_features = np.hstack([business_features, padding])

            
            num_users = len(user_mapping)
            num_businesses = len(business_mapping)
            
            
            x = np.zeros((num_users + num_businesses, feature_dim), dtype=np.float32)
            x[:num_users] = user_features
            
            
            
            if business_features.shape[0] > num_businesses:
                print(f"Warning: Truncating business features from {business_features.shape[0]} to {num_businesses}")
                business_features = business_features[:num_businesses]
            elif business_features.shape[0] < num_businesses:
                print(f"Warning: Padding business features from {business_features.shape[0]} to {num_businesses}")
                padding = np.zeros((num_businesses - business_features.shape[0], feature_dim), dtype=np.float32)
                business_features = np.vstack([business_features, padding])
                
            x[num_users:] = business_features
            
            
            x = torch.FloatTensor(x)
        else:
            x = None
        
        
        data = Data(
            x=x,
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_attr) if edge_attr is not None else None,
            num_nodes=len(self.node_mapping) if include_features else None
        )
        
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(data, os.path.join(cache_dir, 'graph_data.pt'))
            
            
            torch.save({
                'node_mapping': self.node_mapping,
                'reverse_mapping': self.reverse_mapping,
                'node_types': self.node_types
            }, os.path.join(cache_dir, 'mappings.pt'))
        
        return data
    
    def _load_users(self):
        """Load user nodes and features from Neo4j."""
        print("Loading users...")
        features = []
        user_mapping = {}
        
        with self.driver.session() as session:
            
            total_users = session.run("MATCH (u:User) RETURN count(u) AS count").single()['count']
            print(f"Total users: {total_users}")
            
            
            query = """
            MATCH (u:User)
            RETURN u.user_id AS user_id,
                   u.review_count AS review_count,
                   u.average_stars AS average_stars,
                   u.useful_votes AS useful_votes,
                   u.funny_votes AS funny_votes,
                   u.cool_votes AS cool_votes
            SKIP $skip LIMIT $limit
            """
            
            for i in range(0, total_users, self.batch_size):
                result = session.run(query, skip=i, limit=self.batch_size)
                
                for j, record in enumerate(result):
                    user_id = record['user_id']
                    
                    user_mapping[user_id] = i + j
                    
                    
                    feature_vector = [
                        record['review_count'],
                        record['average_stars'],
                        record['useful_votes'],
                        record['funny_votes'],
                        record['cool_votes']
                    ]
                    features.append(feature_vector)
                
                print(f"Loaded {min(i + self.batch_size, total_users)}/{total_users} users")
        
        
        features = np.array(features, dtype=np.float32)
        
        
        if 'user' not in self.scalers:
            self.scalers['user'] = StandardScaler().fit(features)
        features = self.scalers['user'].transform(features)
        
        return features, user_mapping
    
    def _load_businesses(self):
        """Load business nodes and features from Neo4j."""
        print("Loading businesses...")
        features = []
        business_mapping = {}
        
        with self.driver.session() as session:
            
            total_businesses = session.run("MATCH (b:Business) RETURN count(b) AS count").single()['count']
            print(f"Total businesses: {total_businesses}")
            
            
            query = """
            MATCH (b:Business)
            RETURN b.business_id AS business_id,
                   b.stars AS stars,
                   b.review_count AS review_count,
                   b.latitude AS latitude,
                   b.longitude AS longitude,
                   b.is_open AS is_open
            SKIP $skip LIMIT $limit
            """
            
            for i in range(0, total_businesses, self.batch_size):
                result = session.run(query, skip=i, limit=self.batch_size)
                
                for j, record in enumerate(result):
                    business_id = record['business_id']
                    
                    business_mapping[business_id] = i + j
                    
                    
                    feature_vector = [
                        record['stars'],
                        record['review_count'],
                        record['latitude'],
                        record['longitude'],
                        1 if record['is_open'] else 0
                    ]
                    features.append(feature_vector)
                
                print(f"Loaded {min(i + self.batch_size, total_businesses)}/{total_businesses} businesses")
        
        
        features = np.array(features, dtype=np.float32)
        
        
        if 'business' not in self.scalers:
            self.scalers['business'] = StandardScaler().fit(features)
        features = self.scalers['business'].transform(features)
        
        return features, business_mapping
    
    def _load_reviews(self):
        """Load review edges from Neo4j."""
        print("Loading reviews...")
        edge_list = []
        edge_attr_list = []
        
        with self.driver.session() as session:
            
            total_reviews = session.run("MATCH ()-[r:WROTE]->(:Review) RETURN count(r) AS count").single()['count']
            print(f"Total reviews: {total_reviews}")
            
            
            query = """
            MATCH (u:User)-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
            RETURN u.user_id AS user_id,
                   b.business_id AS business_id,
                   r.stars AS stars,
                   r.useful_votes AS useful_votes,
                   r.funny_votes AS funny_votes,
                   r.cool_votes AS cool_votes
            SKIP $skip LIMIT $limit
            """
            
            for i in range(0, total_reviews, self.batch_size):
                result = session.run(query, skip=i, limit=self.batch_size)
                
                for record in result:
                    user_id = record['user_id']
                    business_id = record['business_id']
                    
                    
                    if user_id not in self.node_mapping or business_id not in self.node_mapping:
                        continue
                    
                    
                    user_idx = self.node_mapping[user_id]
                    business_idx = self.node_mapping[business_id]
                    
                    
                    edge_list.append([user_idx, business_idx])
                    
                    edge_list.append([business_idx, user_idx])
                    
                    
                    edge_attr = [record['stars'], record['useful_votes'], record['funny_votes'], record['cool_votes']]
                    edge_attr_list.append(edge_attr)
                    edge_attr_list.append(edge_attr)  
                
                print(f"Loaded {min(i + self.batch_size, total_reviews)}/{total_reviews} reviews")
        
        
        edge_index = np.array(edge_list).T
        edge_attr = np.array(edge_attr_list) if edge_attr_list else None
        
        return edge_index, edge_attr
    
    def create_train_test_split(self, test_size=0.2, negative_sampling_ratio=1.0, cache_dir=None):
        """
        Create training and testing splits for recommendation.
        
        Args:
            test_size: Proportion of data to use for testing
            negative_sampling_ratio: Ratio of negative to positive samples
            cache_dir: Directory to cache splits for faster loading
            
        Returns:
            train_data, val_data, test_data: Training, validation, and test data tensors
        """
        
        if cache_dir:
            cache_path = os.path.join(cache_dir, f'splits_{test_size}_{negative_sampling_ratio}.pt')
            if os.path.exists(cache_path):
                print("Loading cached train/test splits...")
                return torch.load(cache_path)
        
        print("Creating train/test splits...")
        
        
        reviews = self._load_raw_reviews()
        
        
        train_reviews, test_reviews = train_test_split(reviews, test_size=test_size, random_state=42)
        train_reviews, val_reviews = train_test_split(train_reviews, test_size=test_size, random_state=42)
        
        
        train_data = self._create_samples(train_reviews, 1)
        val_data = self._create_samples(val_reviews, 1)
        test_data = self._create_samples(test_reviews, 1)
        
        
        if negative_sampling_ratio > 0:
            train_neg = self._create_negative_samples(train_reviews, ratio=negative_sampling_ratio)
            val_neg = self._create_negative_samples(val_reviews, ratio=negative_sampling_ratio)
            test_neg = self._create_negative_samples(test_reviews, ratio=negative_sampling_ratio)
            
            
            train_data = np.vstack([train_data, train_neg])
            val_data = np.vstack([val_data, val_neg])
            test_data = np.vstack([test_data, test_neg])
        
        
        train_data = torch.LongTensor(train_data)
        val_data = torch.LongTensor(val_data)
        test_data = torch.LongTensor(test_data)
        
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            torch.save((train_data, val_data, test_data), cache_path)
        
        return train_data, val_data, test_data
    
    def _load_raw_reviews(self):
        """Load raw review data for creating train/test splits."""
        reviews = []
        
        with self.driver.session() as session:
            query = """
            MATCH (u:User)-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
            RETURN u.user_id AS user_id,
                   b.business_id AS business_id,
                   r.stars AS stars
            """
            
            result = session.run(query)
            
            for record in result:
                user_id = record['user_id']
                business_id = record['business_id']
                stars = record['stars']
                
                
                if user_id not in self.node_mapping or business_id not in self.node_mapping:
                    continue
                
                user_idx = self.node_mapping[user_id]
                business_idx = self.node_mapping[business_id]
                
                reviews.append((user_idx, business_idx, stars))
        
        return reviews
    
    def _create_samples(self, reviews, label):
        """Create samples from reviews with a given label."""
        samples = []
        
        for user_idx, business_idx, stars in reviews:
            samples.append([user_idx, business_idx, stars, label])
        
        return np.array(samples)
    
    def _create_negative_samples(self, reviews, ratio=1.0):
        """Create negative samples by randomly sampling unseen user-business pairs."""
        
        users = set([user for user, _, _ in reviews])
        businesses = set([business for _, business, _ in reviews])
        
        
        existing_pairs = set([(user, business) for user, business, _ in reviews])
        
        
        num_neg_samples = int(len(reviews) * ratio)
        
        
        neg_samples = []
        users_list = list(users)
        businesses_list = list(businesses)
        
        while len(neg_samples) < num_neg_samples:
            
            user_idx = np.random.choice(users_list)
            business_idx = np.random.choice(businesses_list)
            
            
            if (user_idx, business_idx) not in existing_pairs:
                neg_samples.append([user_idx, business_idx, 0, 0])  
                existing_pairs.add((user_idx, business_idx))  
        
        return np.array(neg_samples)


class YelpDataset(Dataset):
    """
    Memory-efficient dataset for Yelp data that loads data on-demand.
    Used for mini-batch training to handle large graphs.
    """
    def __init__(self, graph_data, split_data, transform=None):
        """
        Initialize the dataset.
        
        Args:
            graph_data: PyTorch Geometric Data object containing the graph
            split_data: PyTorch tensor containing (user_idx, business_idx, stars, label)
            transform: Optional transform to apply to the data
        """
        super(YelpDataset, self).__init__(transform)
        self.graph_data = graph_data
        self.split_data = split_data
    
    def len(self):
        """Return the number of samples in the dataset."""
        return len(self.split_data)
    
    def get(self, idx):
        """Get a sample from the dataset by index."""
        user_idx, business_idx, stars, label = self.split_data[idx]
        
        
        data = Data(
            x=self.graph_data.x,
            edge_index=self.graph_data.edge_index,
            edge_attr=self.graph_data.edge_attr,
            user_idx=user_idx,
            business_idx=business_idx,
            stars=stars,
            label=label
        )
        
        return data