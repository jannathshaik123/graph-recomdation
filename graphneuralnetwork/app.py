import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import torch
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)

# Import custom modules
from data import Neo4jDataLoader
from model import YelpGNN, YelpRecommender
from utils import YelpTrainer

# Neo4j connection configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

# Streamlit app configuration
st.set_page_config(page_title="Yelp Recommendation Explorer", layout="wide")

class RecommendationExplorer:
    def __init__(self):
        # Initialize Neo4j data loader
        self.data_loader = Neo4jDataLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        # Load pre-trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load graph data and model
        self.graph_data = self.data_loader.load_graph_data(cache_dir='cache')
        
        # Initialize and load model
        input_dim = self.graph_data.x.size(1)
        gnn_model = YelpGNN(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=32,
            num_layers=2,
            dropout=0.3,
            gnn_type='sage',
            residual=True,
            batch_norm=True
        )
        self.recommender_model = YelpRecommender(gnn_model)
        
        # Load trainer to use its predict method
        self.trainer = YelpTrainer(self.recommender_model, device=self.device)
        checkpoint_path = os.path.join('checkpoints', 'best_model.pt')
        self.trainer.load_checkpoint(checkpoint_path)
        
        # Mapping between indices and actual IDs
        self.node_mapping = self.data_loader.node_mapping
        self.reverse_mapping = self.data_loader.reverse_mapping
        self.node_types = self.data_loader.node_types
        
        # Fetch additional node information
        self._fetch_node_details()
    
    def _fetch_node_details(self):
        """Fetch user and business names from Neo4j."""
        self.user_details = {}
        self.business_details = {}
        
        with self.data_loader.driver.session() as session:
            # Fetch user details
            user_query = """
            MATCH (u:User)
            RETURN u.user_id AS user_id, u.name AS name
            """
            for record in session.run(user_query):
                self.user_details[record['user_id']] = record['name']
            
            # Fetch business details
            business_query = """
            MATCH (b:Business)
            RETURN b.business_id AS business_id, b.name AS name, b.city AS city
            """
            for record in session.run(business_query):
                self.business_details[record['business_id']] = {
                    'name': record['name'],
                    'city': record['city']
                }
    
    def get_user_name(self, user_id):
        """Get user name from user ID."""
        return self.user_details.get(user_id, user_id)
    
    def get_business_info(self, business_id):
        """Get business name and city from business ID."""
        return self.business_details.get(business_id, {'name': business_id, 'city': 'Unknown'})
    
    def get_user_profile(self, user_id):
        """Fetch detailed user profile from Neo4j."""
        with self.data_loader.driver.session() as session:
            query = """
            MATCH (u:User {user_id: $user_id})
            OPTIONAL MATCH (u)-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
            RETURN u.review_count AS review_count, 
                   u.average_stars AS average_stars,
                   u.useful_votes AS useful_votes,
                   u.funny_votes AS funny_votes,
                   u.cool_votes AS cool_votes,
                   COUNT(DISTINCT b) AS unique_businesses_reviewed
            """
            result = session.run(query, user_id=user_id).single()
            return dict(result) if result else {}
    
    def get_business_profile(self, business_id):
        """Fetch detailed business profile from Neo4j."""
        with self.data_loader.driver.session() as session:
            query = """
            MATCH (b:Business {business_id: $business_id})
            OPTIONAL MATCH (u:User)-[:WROTE]->(r:Review)-[:ABOUT]->(b)
            RETURN b.stars AS avg_stars, 
                   b.review_count AS review_count,
                   b.latitude AS latitude,
                   b.longitude AS longitude,
                   b.is_open AS is_open,
                   b.city AS city,
                   b.state AS state,
                   AVG(r.stars) AS avg_review_stars,
                   COUNT(DISTINCT u) AS total_reviewers
            """
            result = session.run(query, business_id=business_id).single()
            return dict(result) if result else {}
    
    def get_recommendations(self, user_id, top_k=5):
        """Generate recommendations for a user."""
        # Find the user's index in the graph
        user_global_idx = None
        for idx, node_id in self.reverse_mapping.items():
            if node_id == user_id:
                user_global_idx = idx
                break
        
        if user_global_idx is None:
            return []
        
        # Prepare graph data
        x = self.graph_data.x.to(self.device)
        edge_index = self.graph_data.edge_index.to(self.device)
        
        # Get node embeddings
        with torch.no_grad():
            embeddings = self.recommender_model.gnn(x, edge_index)
        
        # Get user embedding
        user_embedding = embeddings[user_global_idx].unsqueeze(0)
        
        # Calculate scores for all businesses
        business_indices = [
            idx for idx, node_type in self.node_types.items() 
            if node_type == 'business'
        ]
        business_embeddings = embeddings[business_indices]
        
        # Compute similarity scores
        similarities = torch.nn.functional.cosine_similarity(
            user_embedding.expand_as(business_embeddings), 
            business_embeddings
        )
        
        # Get top-k recommendations
        top_indices = torch.topk(similarities, top_k).indices
        recommendations = [
            self.reverse_mapping[business_indices[idx]] 
            for idx in top_indices
        ]
        
        return recommendations
    
    def visualize_graph_structure(self):
        """Create a visualization of the graph structure."""
        # Create a NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for idx, node_type in self.node_types.items():
            node_id = self.reverse_mapping[idx]
            if node_type == 'user':
                G.add_node(node_id, type='user')
            else:
                G.add_node(node_id, type='business')
        
        # Add edges from the edge index
        edge_index_np = self.graph_data.edge_index.numpy()
        for i in range(edge_index_np.shape[1]):
            source_idx = edge_index_np[0, i]
            target_idx = edge_index_np[1, i]
            source_id = self.reverse_mapping[source_idx]
            target_id = self.reverse_mapping[target_idx]
            G.add_edge(source_id, target_id)
        
        # Visualize
        plt.figure(figsize=(20, 12))
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Draw nodes
        user_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'user']
        business_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'business']
        
        nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, node_color='lightblue', node_size=20, alpha=0.6, label='Users')
        nx.draw_networkx_nodes(G, pos, nodelist=business_nodes, node_color='lightgreen', node_size=20, alpha=0.6, label='Businesses')
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.1)
        
        plt.title("Yelp Graph Network Structure")
        plt.legend()
        plt.axis('off')
        return plt

def main():
    # Initialize the recommendation explorer
    explorer = RecommendationExplorer()
    
    # Streamlit app
    st.title("🌟 Yelp Recommendation Explorer")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox("Choose a Page", [
        "Graph Visualization", 
        "User Profile Explorer", 
        "Business Profile Explorer", 
        "Recommendation System"
    ])
    
    if page == "Graph Visualization":
        st.header("Graph Network Structure")
        st.write("This visualization shows the network of users and businesses in the Yelp dataset.")
        
        # Create graph visualization
        graph_plt = explorer.visualize_graph_structure()
        st.pyplot(graph_plt)
        
        # Additional graph statistics
        st.subheader("Graph Statistics")
        st.write(f"Total Nodes: {len(explorer.node_types)}")
        st.write(f"Users: {sum(1 for t in explorer.node_types.values() if t == 'user')}")
        st.write(f"Businesses: {sum(1 for t in explorer.node_types.values() if t == 'business')}")
    
    elif page == "User Profile Explorer":
        st.header("User Profile Explorer")
        
        # Get list of users
        users = list(explorer.user_details.keys())
        selected_user_id = st.selectbox("Select a User", users)
        
        if selected_user_id:
            # Fetch user profile
            profile = explorer.get_user_profile(selected_user_id)
            
            # Display user information
            st.subheader(f"Profile for User: {explorer.get_user_name(selected_user_id)}")
            
            # Create columns for profile details
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Review Count", profile.get('review_count', 'N/A'))
                st.metric("Average Stars", f"{profile.get('average_stars', 'N/A'):.2f}")
                st.metric("Unique Businesses Reviewed", profile.get('unique_businesses_reviewed', 'N/A'))
            
            with col2:
                st.metric("Useful Votes", profile.get('useful_votes', 'N/A'))
                st.metric("Funny Votes", profile.get('funny_votes', 'N/A'))
                st.metric("Cool Votes", profile.get('cool_votes', 'N/A'))
    
    elif page == "Business Profile Explorer":
        st.header("Business Profile Explorer")
        
        # Get list of businesses
        businesses = list(explorer.business_details.keys())
        selected_business_id = st.selectbox("Select a Business", businesses)
        
        if selected_business_id:
            # Fetch business profile
            profile = explorer.get_business_profile(selected_business_id)
            business_info = explorer.get_business_info(selected_business_id)
            
            # Display business information
            st.subheader(f"Profile for Business: {business_info['name']}")
            
            # Create columns for profile details
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("City", business_info.get('city', 'N/A'))
                st.metric("Average Business Stars", f"{profile.get('avg_stars', 'N/A'):.2f}")
                st.metric("Review Count", profile.get('review_count', 'N/A'))
                st.metric("Status", "Open" if profile.get('is_open') else "Closed")
            
            with col2:
                st.metric("Total Reviewers", profile.get('total_reviewers', 'N/A'))
                st.metric("Average Review Stars", f"{profile.get('avg_review_stars', 'N/A'):.2f}")
                st.metric("Latitude", f"{profile.get('latitude', 'N/A'):.4f}")
                st.metric("Longitude", f"{profile.get('longitude', 'N/A'):.4f}")
    
    elif page == "Recommendation System":
        st.header("Business Recommendations")
        
        # Get list of users
        users = list(explorer.user_details.keys())
        selected_user_id = st.selectbox("Select a User", users)
        
        if selected_user_id:
            # Get recommendations
            recommendations = explorer.get_recommendations(selected_user_id)
            
            st.subheader(f"Top Recommendations for {explorer.get_user_name(selected_user_id)}")
            
            # Display recommendations
            for i, business_id in enumerate(recommendations, 1):
                business_info = explorer.get_business_info(business_id)
                st.write(f"{i}. {business_info['name']} (City: {business_info['city']})")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("Powered by Graph Neural Network Recommendation System")

if __name__ == "__main__":
    main()