import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data, NeighborSampler, DataLoader
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from neo4j import GraphDatabase
from tqdm import tqdm
import pickle
import random
from collections import defaultdict
import time


class Neo4jDataExtractor:
    """Extracts data from Neo4j database to build GNN training data"""
    
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def get_node_mappings(self):
        """Create mappings from Neo4j IDs to consecutive integers for GNN"""
        print("Extracting node mappings...")
        user_map = {}
        business_map = {}
        
        with self.driver.session() as session:
            # Get users
            result = session.run("MATCH (u:User) RETURN u.user_id AS user_id")
            for i, record in enumerate(result):
                user_map[record["user_id"]] = i
                
            # Get businesses
            result = session.run("MATCH (b:Business) RETURN b.business_id AS business_id")
            for i, record in enumerate(result):
                business_map[record["business_id"]] = i
                
        print(f"Found {len(user_map)} users and {len(business_map)} businesses")
        return user_map, business_map
    
    def get_user_features(self, user_map):
        """Extract user features from Neo4j"""
        print("Extracting user features...")
        user_features = {}
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User) 
                RETURN 
                    u.user_id AS user_id, 
                    u.review_count AS review_count,
                    u.average_stars AS average_stars,
                    u.useful_votes AS useful_votes,
                    u.funny_votes AS funny_votes,
                    u.cool_votes AS cool_votes
            """)
            
            for record in result:
                user_id = record["user_id"]
                if user_id in user_map:
                    internal_id = user_map[user_id]
                    user_features[internal_id] = [
                        float(record["review_count"]),
                        float(record["average_stars"]),
                        float(record["useful_votes"]),
                        float(record["funny_votes"]),
                        float(record["cool_votes"])
                    ]
        
        # Convert to tensor with proper shape
        max_id = max(user_features.keys()) + 1
        feature_tensor = torch.zeros((max_id, 5))
        for uid, features in user_features.items():
            feature_tensor[uid] = torch.tensor(features)
            
        # Normalize features
        feature_means = torch.mean(feature_tensor, dim=0)
        feature_stds = torch.std(feature_tensor, dim=0)
        feature_stds[feature_stds == 0] = 1  # Avoid division by zero
        normalized_features = (feature_tensor - feature_means) / feature_stds
            
        return normalized_features
    
    def get_business_features(self, business_map):
        """Extract business features from Neo4j"""
        print("Extracting business features...")
        business_features = {}
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (b:Business) 
                RETURN 
                    b.business_id AS business_id, 
                    b.stars AS stars,
                    b.review_count AS review_count,
                    b.latitude AS latitude,
                    b.longitude AS longitude
            """)
            
            for record in result:
                business_id = record["business_id"]
                if business_id in business_map:
                    internal_id = business_map[business_id]
                    business_features[internal_id] = [
                        float(record["stars"]),
                        float(record["review_count"]),
                        float(record["latitude"]),
                        float(record["longitude"])
                    ]
        
        # Get category information as one-hot encoding
        category_map = {}
        result = session.run("""
            MATCH (c:Category)<-[:IN_CATEGORY]-(b:Business) 
            RETURN DISTINCT c.name AS category
        """)
        
        for i, record in enumerate(result):
            category_map[record["category"]] = i
            
        print(f"Found {len(category_map)} unique categories")
        
        # Get business-category relationships
        business_categories = defaultdict(list)
        result = session.run("""
            MATCH (b:Business)-[:IN_CATEGORY]->(c:Category)
            RETURN b.business_id AS business_id, c.name AS category
        """)
        
        for record in result:
            business_id = record["business_id"]
            category = record["category"]
            if business_id in business_map and category in category_map:
                internal_id = business_map[business_id]
                category_id = category_map[category]
                business_categories[internal_id].append(category_id)
        
        # Convert to tensor with proper shape
        max_id = max(business_features.keys()) + 1
        feature_tensor = torch.zeros((max_id, 4))
        for bid, features in business_features.items():
            feature_tensor[bid] = torch.tensor(features)
            
        # Normalize features
        feature_means = torch.mean(feature_tensor, dim=0)
        feature_stds = torch.std(feature_tensor, dim=0)
        feature_stds[feature_stds == 0] = 1  # Avoid division by zero
        normalized_features = (feature_tensor - feature_means) / feature_stds
            
        # We'll use the category information as additional features later
        return normalized_features, business_categories, category_map
    
    def get_interactions(self, user_map, business_map):
        """Extract user-business interactions (reviews)"""
        print("Extracting user-business interactions...")
        interactions = []
        ratings = {}
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User)-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
                RETURN u.user_id AS user_id, b.business_id AS business_id, r.stars AS rating
            """)
            
            for record in result:
                user_id = record["user_id"]
                business_id = record["business_id"]
                rating = float(record["rating"])
                
                if user_id in user_map and business_id in business_map:
                    u_idx = user_map[user_id]
                    b_idx = business_map[business_id]
                    interactions.append((u_idx, b_idx))
                    ratings[(u_idx, b_idx)] = rating
                    
        print(f"Found {len(interactions)} interactions")
        return interactions, ratings


class GraphDataset:
    """Prepares graph data for GNN training"""
    
    def __init__(self, user_features, business_features, interactions, ratings, 
                 business_categories=None, category_map=None, test_ratio=0.2):
        self.user_features = user_features
        self.business_features = business_features
        self.interactions = interactions
        self.ratings = ratings
        self.business_categories = business_categories
        self.category_map = category_map
        
        # Split the data into train and test sets
        self.train_interactions, self.test_interactions = train_test_split(
            interactions, test_size=test_ratio, random_state=42
        )
        
        # Create train and test masks
        self.train_mask = {(u, b): True for u, b in self.train_interactions}
        self.test_mask = {(u, b): True for u, b in self.test_interactions}
        
        # Create edge index for PyTorch Geometric
        self._create_edge_index()
        
    def _create_edge_index(self):
        """Create edge index for PyTorch Geometric from interactions"""
        user_nodes = [u for u, _ in self.interactions]
        business_nodes = [b for _, b in self.interactions]
        
        # Determine the total number of nodes
        num_users = self.user_features.size(0)
        num_businesses = self.business_features.size(0)
        
        # Create edge index
        edge_index = torch.tensor([user_nodes + business_nodes, 
                                 business_nodes + user_nodes], dtype=torch.long)
        
        # Create edge features (ratings)
        edge_attr = []
        for u, b in zip(user_nodes, business_nodes):
            edge_attr.append(self.ratings.get((u, b), 0))
        for b, u in zip(business_nodes, user_nodes):
            edge_attr.append(self.ratings.get((u, b), 0))
            
        self.edge_index = edge_index
        self.edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
        
    def create_pytorch_data(self):
        """Create PyTorch Geometric Data object"""
        # Combine user and business features
        x = torch.cat([self.user_features, self.business_features], dim=0)
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr)
        
        return data
    
    def get_train_samples(self, negative_ratio=4):
        """Get training samples with negative sampling"""
        positive_samples = [(u, b, self.ratings[(u, b)]) for u, b in self.train_interactions]
        
        # Generate negative samples
        negative_samples = []
        for _ in range(negative_ratio * len(positive_samples)):
            u = random.choice(range(self.user_features.size(0)))
            b = random.choice(range(self.business_features.size(0)))
            if (u, b) not in self.train_mask and (u, b) not in self.test_mask:
                negative_samples.append((u, b, 0.0))
                
        return positive_samples + negative_samples
    
    def get_test_samples(self):
        """Get test samples"""
        return [(u, b, self.ratings[(u, b)]) for u, b in self.test_interactions]


class GNNRecommender(nn.Module):
    """Graph Neural Network for recommendation"""
    
    def __init__(self, user_feature_dim, business_feature_dim, hidden_dims, dropout=0.2):
        super(GNNRecommender, self).__init__()
        
        # User and business feature projections
        self.user_proj = nn.Linear(user_feature_dim, hidden_dims[0])
        self.business_proj = nn.Linear(business_feature_dim, hidden_dims[0])
        
        # Graph convolution layers
        self.conv1 = GCNConv(hidden_dims[0], hidden_dims[1])
        self.conv2 = GCNConv(hidden_dims[1], hidden_dims[2])
        self.conv3 = GCNConv(hidden_dims[2], hidden_dims[3])
        
        # Prediction layer
        self.pred = nn.Sequential(
            nn.Linear(2 * hidden_dims[3], hidden_dims[3]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[3], 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Split features for users and businesses
        num_users = self.user_proj.in_features
        user_x = x[:num_users]
        business_x = x[num_users:]
        
        # Project features
        user_h = self.user_proj(user_x)
        business_h = self.business_proj(business_x)
        
        # Combine features
        x = torch.cat([user_h, business_h], dim=0)
        
        # Apply graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        
        return x
    
    def predict(self, user_idx, business_idx, user_emb, business_emb):
        # Get embeddings for specific users and businesses
        user_h = user_emb[user_idx]
        business_h = business_emb[business_idx]
        
        # Concatenate embeddings
        h = torch.cat([user_h, business_h], dim=1)
        
        # Apply prediction layer
        pred = self.pred(h)
        
        return pred.squeeze()


# Memory-efficient mini-batch training with neighbor sampling
class MiniBatchTrainer:
    """Memory-efficient mini-batch training with neighbor sampling"""
    
    def __init__(self, model, data, device):
        self.model = model
        self.data = data
        self.device = device
        self.num_users = model.user_proj.in_features
        
    def create_sampler(self, batch_size, num_hops=2):
        """Create neighbor sampler for mini-batch training"""
        train_loader = NeighborSampler(
            self.data.edge_index, 
            node_idx=None,
            sizes=[15, 10],  # Number of neighbors to sample for each node
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        return train_loader
    
    def train_epoch(self, train_samples, optimizer, batch_size=1024):
        """Train for one epoch using mini-batches"""
        self.model.train()
        
        # Create data loader for training samples
        train_loader = DataLoader(train_samples, batch_size=batch_size, shuffle=True)
        
        total_loss = 0
        
        for batch in train_loader:
            user_idx, business_idx, ratings = batch
            
            # Forward pass to get embeddings for all nodes
            node_embeddings = self.model(self.data)
            user_emb = node_embeddings[:self.num_users]
            business_emb = node_embeddings[self.num_users:]
            
            # Get predictions for specific user-business pairs
            pred = self.model.predict(user_idx, business_idx, user_emb, business_emb)
            
            # Calculate loss
            loss = F.mse_loss(pred, ratings)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def evaluate(self, test_samples, batch_size=1024):
        """Evaluate model on test samples"""
        self.model.eval()
        
        # Create data loader for test samples
        test_loader = DataLoader(test_samples, batch_size=batch_size, shuffle=False)
        
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            # Get embeddings for all nodes
            node_embeddings = self.model(self.data)
            user_emb = node_embeddings[:self.num_users]
            business_emb = node_embeddings[self.num_users:]
            
            for batch in test_loader:
                user_idx, business_idx, ratings = batch
                
                # Get predictions for specific user-business pairs
                pred = self.model.predict(user_idx, business_idx, user_emb, business_emb)
                
                # Calculate loss
                loss = F.mse_loss(pred, ratings)
                total_loss += loss.item()
                
                predictions.extend(pred.cpu().numpy())
                targets.extend(ratings.cpu().numpy())
                
        # Calculate metrics
        mse = np.mean((np.array(predictions) - np.array(targets)) ** 2)
        rmse = np.sqrt(mse)
        
        return {
            'loss': total_loss / len(test_loader),
            'mse': mse,
            'rmse': rmse
        }


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0.001, path='best_model.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class YelpGNNRecommendationSystem:
    """Complete recommendation system using GNN"""
    
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, device='cuda'):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.user_map = None
        self.business_map = None
        
    def load_data(self, cache_dir='./cache'):
        """Load data from Neo4j or cache"""
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, 'gnn_data.pkl')
        
        if os.path.exists(cache_file):
            print(f"Loading data from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                self.user_map = data['user_map']
                self.business_map = data['business_map']
                self.user_features = data['user_features']
                self.business_features = data['business_features']
                self.interactions = data['interactions']
                self.ratings = data['ratings']
                self.business_categories = data['business_categories']
                self.category_map = data['category_map']
        else:
            print("Extracting data from Neo4j...")
            extractor = Neo4jDataExtractor(self.neo4j_uri, self.neo4j_user, self.neo4j_password)
            
            try:
                # Get node mappings
                self.user_map, self.business_map = extractor.get_node_mappings()
                
                # Get features
                self.user_features = extractor.get_user_features(self.user_map)
                self.business_features, self.business_categories, self.category_map = extractor.get_business_features(self.business_map)
                
                # Get interactions
                self.interactions, self.ratings = extractor.get_interactions(self.user_map, self.business_map)
                
                # Save to cache
                data = {
                    'user_map': self.user_map,
                    'business_map': self.business_map,
                    'user_features': self.user_features,
                    'business_features': self.business_features,
                    'interactions': self.interactions,
                    'ratings': self.ratings,
                    'business_categories': self.business_categories,
                    'category_map': self.category_map
                }
                
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
                    
            finally:
                extractor.close()
                
        # Create graph dataset
        self.dataset = GraphDataset(
            self.user_features, 
            self.business_features, 
            self.interactions, 
            self.ratings,
            self.business_categories,
            self.category_map
        )
        
        # Create PyTorch Geometric data object
        self.data = self.dataset.create_pytorch_data().to(self.device)
        
        print(f"Data loaded: {len(self.user_map)} users, {len(self.business_map)} businesses, {len(self.interactions)} interactions")
                
    def build_model(self, hidden_dims=[64, 128, 64, 32], dropout=0.3):
        """Build the GNN model"""
        user_feature_dim = self.user_features.size(1)
        business_feature_dim = self.business_features.size(1)
        
        self.model = GNNRecommender(
            user_feature_dim, 
            business_feature_dim, 
            hidden_dims, 
            dropout
        ).to(self.device)
        
        print(f"Model built with {sum(p.numel() for p in self.model.parameters())} parameters")
        
    def train(self, epochs=50, batch_size=1024, lr=0.001, weight_decay=1e-5):
        """Train the model"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        early_stopping = EarlyStopping(patience=10, path='best_gnn_model.pt')
        
        # Get training and test samples
        train_samples = self.dataset.get_train_samples()
        test_samples = self.dataset.get_test_samples()
        
        # Convert to tensors
        train_data = [
            torch.tensor([u for u, _, _ in train_samples], dtype=torch.long).to(self.device),
            torch.tensor([b for _, b, _ in train_samples], dtype=torch.long).to(self.device),
            torch.tensor([r for _, _, r in train_samples], dtype=torch.float).to(self.device)
        ]
        
        test_data = [
            torch.tensor([u for u, _, _ in test_samples], dtype=torch.long).to(self.device),
            torch.tensor([b for _, b, _ in test_samples], dtype=torch.long).to(self.device),
            torch.tensor([r for _, _, r in test_samples], dtype=torch.float).to(self.device)
        ]
        
        # Create trainer
        trainer = MiniBatchTrainer(self.model, self.data, self.device)
        
        # Train
        print("Starting training...")
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train for one epoch
            train_loss = trainer.train_epoch(train_data, optimizer, batch_size)
            
            # Evaluate
            eval_metrics = trainer.evaluate(test_data, batch_size)
            
            # Update learning rate
            scheduler.step(eval_metrics['loss'])
            
            # Print metrics
            time_taken = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Test Loss: {eval_metrics['loss']:.4f}, "
                  f"RMSE: {eval_metrics['rmse']:.4f}, "
                  f"Time: {time_taken:.2f}s")
            
            # Check for early stopping
            if early_stopping(eval_metrics['loss'], self.model):
                print("Early stopping triggered")
                break
                
        # Load best model
        self.model.load_state_dict(torch.load('best_gnn_model.pt'))
        print("Training completed")
        
    def save_model(self, path='yelp_gnn_recommender.pt'):
        """Save the trained model"""
        if self.model is not None:
            # Save model state dict
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'user_map': self.user_map,
                'business_map': self.business_map,
                'user_features': self.user_features,
                'business_features': self.business_features
            }, path)
            print(f"Model saved to {path}")
        else:
            print("No model to save")
            
    def load_model(self, path='yelp_gnn_recommender.pt'):
        """Load a trained model"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load maps and features
            self.user_map = checkpoint['user_map']
            self.business_map = checkpoint['business_map']
            self.user_features = checkpoint['user_features']
            self.business_features = checkpoint['business_features']
            
            # Build model
            self.build_model()
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {path}")
            return True
        else:
            print(f"Model file not found: {path}")
            return False
            
    def get_recommendations(self, user_id, top_k=10):
        """Get recommendations for a user"""
        if self.model is None:
            print("Model not loaded")
            return []
            
        if user_id not in self.user_map:
            print(f"User {user_id} not found")
            return []
            
        self.model.eval()
        user_idx = self.user_map[user_id]
        
        # Get all businesses
        business_indices = list(range(len(self.business_map)))
        
        # Create tensors
        user_indices = torch.tensor([user_idx] * len(business_indices), dtype=torch.long).to(self.device)
        business_indices = torch.tensor(business_indices, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            # Get embeddings for all nodes
            node_embeddings = self.model(self.data)
            user_emb = node_embeddings[:self.user_features.size(0)]
            business_emb = node_embeddings[self.user_features.size(0):]
            
            # Get predictions for user-business pairs
            predictions = self.model.predict(user_indices, business_indices, user_emb, business_emb)
            
        # Get top-k businesses
        _, indices = torch.topk(predictions, top_k)
        top_businesses = [list(self.business_map.keys())[i] for i in indices.cpu().numpy()]
        
        return top_businesses


# Usage example
def main():
    # Neo4j connection information
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"  # Change to your actual password
    
    # Create recommendation system
    recommender = YelpGNNRecommendationSystem(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # Load data
    recommender.load_data()
    
    # Build model
    recommender.build_model()
    
    # Train model
    recommender.train(epochs=30, batch_size=2048)
    
    # Save model
    recommender.save_model()
    
    # Example: Get recommendations for a user
    user_id = list(recommender.user_map.keys())[0]  # Get first user as example
    recommendations = recommender.get_recommendations(user_id, top_k=10)
    print(f"Top recommendations for user {user_id}:")
    for i, biz_id in enumerate(recommendations):
        print(f"{i+1}. {biz_id}")


if __name__ == "__main__":
    main()
