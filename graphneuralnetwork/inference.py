import os
import argparse
import torch
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from model import YelpGNN, YelpRecommender
from data import Neo4jDataLoader

# Neo4j connection configuration
NEO4J_URI = "bolt://localhost:7687"  # Update if your Neo4j instance is hosted elsewhere
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # Change this to your actual password

class YelpRecommendationInference:
    """
    Inference class for Yelp GNN recommendation system.
    Used to generate recommendations for specific users.
    """
    def __init__(self, model_path, cache_dir='cache', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the trained model checkpoint
            cache_dir: Directory containing cached data
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.cache_dir = cache_dir
        self.model_path = model_path
        
        # Create cache directory if needed
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Initialize data loader
        self.data_loader = Neo4jDataLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        # Load node mappings
        self.mappings = torch.load(os.path.join(cache_dir, 'mappings.pt'))
        self.node_mapping = self.mappings['node_mapping']
        self.reverse_mapping = self.mappings['reverse_mapping']
        self.node_types = self.mappings['node_types']
        
        # Load graph data
        self.graph_data = torch.load(os.path.join(cache_dir, 'graph_data.pt'))
        self.graph_data = self.graph_data.to(device)
        
        # Load model
        self.model = self._load_model()
        self.model.to(device)
        self.model.eval()
        
        print(f"Inference engine initialized on {device}")
        
    def _load_model(self):
        """Load the trained model from checkpoint."""
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model parameters
        if 'model_params' in checkpoint:
            model_params = checkpoint['model_params']
        else:
            # Default parameters if not provided in checkpoint
            model_params = {
                'input_dim': self.graph_data.x.size(1),
                'hidden_dim': 64,
                'output_dim': 32,
                'num_layers': 2,
                'dropout': 0.3,
                'gnn_type': 'sage',
                'residual': True,
                'batch_norm': True
            }
        
        # Initialize model
        gnn_model = YelpGNN(
            input_dim=model_params['input_dim'],
            hidden_dim=model_params['hidden_dim'],
            output_dim=model_params['output_dim'],
            num_layers=model_params['num_layers'],
            dropout=model_params['dropout'],
            gnn_type=model_params['gnn_type'],
            residual=model_params['residual'],
            batch_norm=model_params['batch_norm']
        )
        
        model = YelpRecommender(gnn_model)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {self.model_path}")
        
        return model
    
    def get_recommendations(self, user_id, top_k=10, filter_rated=True):
        """
        Get recommendations for a specific user.
        
        Args:
            user_id: Neo4j ID of the user to recommend for
            top_k: Number of recommendations to return
            filter_rated: Whether to filter out businesses already rated by the user
            
        Returns:
            DataFrame containing recommended businesses with scores
        """
        # Check if user exists in mapping
        if user_id not in self.node_mapping:
            raise ValueError(f"User ID {user_id} not found in the graph")
        
        # Get user index
        user_idx = self.node_mapping[user_id]
        
        # Get already rated businesses if needed
        rated_businesses = set()
        if filter_rated:
            with self.data_loader.driver.session() as session:
                query = """
                MATCH (u:User {user_id: $user_id})-[:WROTE]->(:Review)-[:ABOUT]->(b:Business)
                RETURN b.business_id AS business_id
                """
                result = session.run(query, user_id=user_id)
                rated_businesses = {record['business_id'] for record in result}
        
        # Get all business indices
        business_indices = [idx for idx, node_type in self.node_types.items() 
                           if node_type == 'business']
        
        # Prepare user-business pairs for scoring
        user_indices = [user_idx] * len(business_indices)
        
        # Forward pass to get embeddings
        with torch.no_grad():
            # Get embeddings
            embeddings = self.model.gnn(self.graph_data.x, self.graph_data.edge_index)
            
            # Get user embedding
            user_embed = embeddings[user_idx]
            
            # Get business embeddings
            business_embeds = embeddings[business_indices]
            
            # Compute scores in batches to avoid OOM
            batch_size = 1024
            all_scores = []
            
            for i in range(0, len(business_indices), batch_size):
                # Get batch of business embeddings
                batch_embeds = business_embeds[i:i+batch_size]
                
                # Replicate user embedding for batch
                batch_user_embeds = user_embed.unsqueeze(0).expand(batch_embeds.size(0), -1)
                
                # Concatenate user and business embeddings
                pair_embeds = torch.cat([batch_user_embeds, batch_embeds], dim=1)
                
                # Predict scores
                batch_scores = self.model.predictor(pair_embeds).squeeze()
                all_scores.append(batch_scores)
            
            # Combine all batches
            scores = torch.cat(all_scores).cpu().numpy()
        
        # Create recommendations DataFrame
        recommendations = []
        for i, business_idx in enumerate(business_indices):
            business_id = self.reverse_mapping[business_idx]
            
            # Skip already rated businesses if filtering
            if filter_rated and business_id in rated_businesses:
                continue
                
            recommendations.append({
                'business_id': business_id,
                'score': scores[i]
            })
        
        # Convert to DataFrame
        df_recommendations = pd.DataFrame(recommendations)
        
        # Sort by score and get top-k
        df_recommendations = df_recommendations.sort_values('score', ascending=False).head(top_k)
        
        # Optionally fetch additional business information
        df_recommendations = self._enrich_business_data(df_recommendations)
        
        return df_recommendations
    
    def _enrich_business_data(self, df_recommendations):
        """Add additional business information to recommendations."""
        business_ids = df_recommendations['business_id'].tolist()
        
        # Fetch business data from Neo4j
        with self.data_loader.driver.session() as session:
            query = """
            MATCH (b:Business)
            WHERE b.business_id IN $business_ids
            RETURN b.business_id AS business_id,
                   b.name AS name,
                   b.stars AS stars,
                   b.review_count AS review_count,
                   b.categories AS categories
            """
            result = session.run(query, business_ids=business_ids)
            
            # Create business info dictionary
            business_info = {}
            for record in result:
                business_info[record['business_id']] = {
                    'name': record['name'],
                    'stars': record['stars'],
                    'review_count': record['review_count'],
                    'categories': record['categories']
                }
        
        # Add business info to recommendations
        enriched_data = []
        for _, row in df_recommendations.iterrows():
            business_id = row['business_id']
            data = {
                'business_id': business_id,
                'predicted_score': row['score']
            }
            
            # Add additional info if available
            if business_id in business_info:
                data.update(business_info[business_id])
                
            enriched_data.append(data)
            
        return pd.DataFrame(enriched_data)
    
    def get_similar_businesses(self, business_id, top_k=10):
        """
        Find similar businesses based on embeddings.
        
        Args:
            business_id: Neo4j ID of the business to find similar ones for
            top_k: Number of similar businesses to return
            
        Returns:
            DataFrame containing similar businesses with similarity scores
        """
        # Check if business exists in mapping
        if business_id not in self.node_mapping:
            raise ValueError(f"Business ID {business_id} not found in the graph")
        
        # Get business index
        business_idx = self.node_mapping[business_id]
        
        # Get all business indices except the query business
        business_indices = [idx for idx, node_type in self.node_types.items() 
                           if node_type == 'business' and idx != business_idx]
        
        # Forward pass to get embeddings
        with torch.no_grad():
            # Get embeddings
            embeddings = self.model.gnn(self.graph_data.x, self.graph_data.edge_index)
            
            # Get query business embedding
            query_embed = embeddings[business_idx]
            
            # Get other business embeddings
            business_embeds = embeddings[business_indices]
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(query_embed.unsqueeze(0), business_embeds)
            similarity = similarity.cpu().numpy()
        
        # Create recommendations DataFrame
        similar_businesses = []
        for i, other_idx in enumerate(business_indices):
            other_id = self.reverse_mapping[other_idx]
            similar_businesses.append({
                'business_id': other_id,
                'similarity': similarity[i]
            })
        
        # Convert to DataFrame
        df_similar = pd.DataFrame(similar_businesses)
        
        # Sort by similarity and get top-k
        df_similar = df_similar.sort_values('similarity', ascending=False).head(top_k)
        
        # Enrich with business data
        df_similar = self._enrich_business_data(df_similar)
        df_similar = df_similar.rename(columns={'predicted_score': 'similarity'})
        
        return df_similar
    
    def close(self):
        """Close resources."""
        if hasattr(self, 'data_loader'):
            self.data_loader.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Yelp GNN Recommendation Inference')
    
    # Required parameters
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the trained model checkpoint')
    
    # Optional parameters
    parser.add_argument('--cache_dir', type=str, default='cache',
                        help='Directory containing cached data')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of recommendations to generate')
    parser.add_argument('--user_id', type=str, default=None,
                        help='User ID to generate recommendations for')
    parser.add_argument('--business_id', type=str, default=None,
                        help='Business ID to find similar businesses for')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save recommendations as CSV')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Initialize inference engine
    inference = YelpRecommendationInference(
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        device=args.device
    )
    
    try:
        # Generate recommendations
        if args.user_id:
            print(f"Generating recommendations for user {args.user_id}")
            recommendations = inference.get_recommendations(
                user_id=args.user_id,
                top_k=args.top_k
            )
            print(f"Top {len(recommendations)} recommendations:")
            print(recommendations)
            
            # Save to CSV if requested
            if args.output:
                recommendations.to_csv(args.output, index=False)
                print(f"Recommendations saved to {args.output}")
        
        # Find similar businesses
        elif args.business_id:
            print(f"Finding similar businesses to {args.business_id}")
            similar = inference.get_similar_businesses(
                business_id=args.business_id,
                top_k=args.top_k
            )
            print(f"Top {len(similar)} similar businesses:")
            print(similar)
            
            # Save to CSV if requested
            if args.output:
                similar.to_csv(args.output, index=False)
                print(f"Similar businesses saved to {args.output}")
        
        else:
            print("Please provide either --user_id or --business_id")
    
    finally:
        # Close resources
        inference.close()


if __name__ == '__main__':
    main()