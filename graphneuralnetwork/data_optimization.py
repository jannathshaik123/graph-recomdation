import numpy as np
import torch
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from sklearn.decomposition import PCA
from neo4j import GraphDatabase
import random
from collections import defaultdict
import os
import pickle
from tqdm import tqdm


class FeatureProcessor:
    """Process and compress features for memory efficiency"""
    
    def __init__(self, method='pca', n_components=16):
        self.method = method
        self.n_components = n_components
        self.models = {}
        
    def fit_transform(self, features, feature_type):
        """Fit and transform features"""
        if self.method == 'pca':
            # Initialize PCA model if not exist
            if feature_type not in self.models:
                self.models[feature_type] = PCA(n_components=self.n_components)
                
            # Apply PCA
            transformed = self.models[feature_type].fit_transform(features.cpu().numpy())
            return torch.tensor(transformed, dtype=torch.float)
        
        elif self.method == 'quantile':
            # Simple feature binning for memory reduction
            bins = 10
            transformed = []
            
            for i in range(features.shape[1]):
                feature = features[:, i].cpu().numpy()
                quantiles = np.percentile(feature, np.linspace(0, 100, bins + 1))
                binned = np.digitize(feature, quantiles) - 1
                binned = np.clip(binned, 0, bins - 1) / (bins - 1)  # Normalize to [0, 1]
                transformed.append(binned)
                
            return torch.tensor(np.column_stack(transformed), dtype=torch.float)
        
        elif self.method == 'sampling':
            # Return a subset of features
            if features.shape[1] <= self.n_components:
                return features
                
            # Randomly select features
            selected = random.sample(range(features.shape[1]), self.n_components)
            return features[:, selected]
        
        else:
            # No transformation
            return features
            
    def transform(self, features, feature_type):
        """Transform features using fitted model"""
        if self.method == 'pca' and feature_type in self.models:
            transformed = self.models[feature_type].transform(features.cpu().numpy())
            return torch.tensor(transformed, dtype=torch.float)
        else:
            # Fall back to fit_transform
            return self.fit_transform(features, feature_type)


class GraphDataProcessor:
    """Process graph data for memory-efficient training"""
    
    def __init__(self, uri, user, password, cache_dir='./cache'):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def close(self):
        self.driver.close()
        
    def process_graph(self, use_cache=True):
        """Process graph data from Neo4j"""
        cache_file = os.path.join(self.cache_dir, 'processed_graph.pkl')
        
        if use_cache and os.path.exists(cache_file):
            print(f"Loading processed graph from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print("Processing graph data...")
        
        # Extract nodes and mappings
        user_map, business_map = self._extract_node_mappings()
        
        # Extract features
        user_features = self._extract_user_features(user_map)
        business_features = self._extract_business_features(business_map)
        
        # Compress features for memory efficiency
        processor = FeatureProcessor(method='pca', n_components=min(16, user_features.shape[1]))
        user_features = processor.fit_transform(user_features, 'user')
        business_features = processor.fit_transform(business_features, 'business')
        
        # Extract interactions
        interactions, ratings = self._extract_interactions(user_map, business_map)
        
        # Create edge index
        edge_index, edge_attr = self._create_edge_index(interactions, ratings)
        
        # Create PyTorch Geometric data object
        data = Data(
            x=torch.cat([user_features, business_features], dim=0),
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        # Save to cache
        result = {
            'data': data,
            'user_map': user_map,
            'business_map': business_map,
            'user_features': user_features,
            'business_features': business_features,
            'interactions': interactions,
            'ratings': ratings
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
            
        return result
        
    def _extract_node_mappings(self):
        """Extract node mappings from Neo4j"""
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
    
    def _extract_user_features(self, user_map):
        """Extract user features from Neo4j"""
        print("Extracting user features...")
        features = []
        
        with self.driver.session() as session:
            # Get basic user features
            result = session.run("""
                MATCH (u:User) 
                RETURN 
                    u.user_id AS user_id, 
                    u.review_count AS review_count,
                    u.average_stars AS average_stars,
                    u.useful_votes AS useful_votes,
                    u.funny_votes AS funny_votes,
                    u.cool_votes AS cool_votes
                ORDER BY id(u)
            """)
            
            # Create a tensor with zeros
            max_id = max(user_map.values()) + 1
            feature_tensor = torch.zeros((max_id, 5))
            
            for record in result:
                user_id = record["user_id"]
                if user_id in user_map:
                    idx = user_map[user_id]
                    feature_tensor[idx] = torch.tensor([
                        float(record["review_count"]),
                        float(record["average_stars"]),
                        float(record["useful_votes"]),
                        float(record["funny_votes"]),
                        float(record["cool_votes"])
                    ])
            
            # Normalize features
            means = torch.mean(feature_tensor, dim=0)
            stds = torch.std(feature_tensor, dim=0)
            stds[stds == 0] = 1  # Avoid division by zero
            normalized_tensor = (feature_tensor - means) / stds
            
            return normalized_tensor
    
    def _extract_business_features(self, business_map):
        """Extract business features from Neo4j"""
        print("Extracting business features...")
        
        with self.driver.session() as session:
            # Get business features
            result = session.run("""
                MATCH (b:Business) 
                RETURN 
                    b.business_id AS business_id, 
                    b.stars AS stars,
                    b.review_count AS review_count,
                    b.latitude AS latitude,
                    b.longitude AS longitude
                ORDER BY id(b)
            """)
            
            # Create a tensor with zeros
            max_id = max(business_map.values()) + 1
            feature_tensor = torch.zeros((max_id, 4))
            
            for record in result:
                business_id = record["business_id"]
                if business_id in business_map:
                    idx = business_map[business_id]
                    feature_tensor[idx] = torch.tensor([
                        float(record["stars"]),
                        float(record["review_count"]),
                        float(record["latitude"]),
                        float(record["longitude"])
                    ])
            
            # Normalize features
            means = torch.mean(feature_tensor, dim=0)
            stds = torch.std(feature_tensor, dim=0)
            stds[stds == 0] = 1  # Avoid division by zero
            normalized_tensor = (feature_tensor - means) / stds
            
            return normalized_tensor
    
    def _extract_interactions(self, user_map, business_map):
        """Extract user-business interactions from Neo4j"""
        print("Extracting interaction data...")
        interactions = []
        ratings = {}
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User)-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
                RETURN u.user_id AS user_id, b.business_id AS business_id, r.stars AS stars
            """)
            
            for record in result:
                user_id = record["user_id"]
                business_id = record["business_id"]
                
                if user_id in user_map and business_id in business_map:
                    user_idx = user_map[user_id]
                    business_idx = business_map[business_id]
                    
                    interactions.append((user_idx, business_idx))
                    ratings[(user_idx, business_idx)] = float(record["stars"])
        
        print(f"Found {len(interactions)} interactions")
        return interactions, ratings
    
    def _create_edge_index(self, interactions, ratings):
        """Create edge index and attributes for PyTorch Geometric"""
        print("Creating edge index...")
        
        # Extract source and target nodes
        src_nodes = [u for u, _ in interactions]
        dst_nodes = [b for _, b in interactions]
        
        # Create bidirectional edge index
        edge_index = torch.tensor([src_nodes + dst_nodes, dst_nodes + src_nodes], dtype=torch.long)
        
        # Create edge attributes (ratings)
        edge_attr = []
        for u, b in interactions:
            edge_attr.append(ratings[(u, b)])
        # Add reversed edges with same ratings
        for u, b in interactions:
            edge_attr.append(ratings[(u, b)])
            
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
        
        return edge_index, edge_attr
    
    def extract_subgraph(self, node_indices, hops=2):
        """Extract a subgraph around specific nodes for memory efficiency"""
        # TODO: Implement subgraph extraction for more efficient processing
        pass
    
    def create_sparse_matrix(self, edge_index, num_nodes):
        """Create sparse adjacency matrix for more efficient computation"""
        adj_t = SparseTensor(
            row=edge_index[0], 
            col=edge_index[1],
            sparse_sizes=(num_nodes, num_nodes)
        )
        return adj_t
    
    def prune_low_degree_nodes(self, edge_index, min_degree=2):
        """Remove nodes with degree less than min_degree to reduce noise"""
        # Count node degrees
        degrees = defaultdict(int)
        for i in range(edge_index.shape[1]):
            degrees[edge_index[0, i].item()] += 1
            
        # Filter edges
        keep_mask = []
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            if degrees[src] >= min_degree and degrees[dst] >= min_degree:
                keep_mask.append(True)
            else:
                keep_mask.append(False)
                
        # Apply mask
        keep_mask = torch.tensor(keep_mask, dtype=torch.bool)
        pruned_edge_index = edge_index[:, keep_mask]
        
        return pruned_edge_index
    
    def precompute_node_embeddings(self, model, data, batch_size=1024):
        """Precompute and cache node embeddings for faster inference"""
        model.eval()
        num_nodes = data.x.size(0)
        embeddings = torch.zeros((num_nodes, model.out_channels), 
                                device=data.x.device)
        
        with torch.no_grad():
            for i in tqdm(range(0, num_nodes, batch_size)):
                end_idx = min(i + batch_size, num_nodes)
                batch_x = data.x[i:end_idx]
                batch_emb = model(batch_x, data.edge_index)
                embeddings[i:end_idx] = batch_emb
                
        return embeddings
