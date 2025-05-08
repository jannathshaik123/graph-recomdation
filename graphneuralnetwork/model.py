import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class YelpGNN(nn.Module):
    """
    Graph Neural Network model for Yelp recommendation system.
    Supports multiple GNN layer types and includes techniques to prevent overfitting.
    """
    def __init__(self, 
                 input_dim, 
                 hidden_dim=64, 
                 output_dim=32, 
                 num_layers=2,
                 dropout=0.3, 
                 gnn_type='sage',
                 residual=True,
                 batch_norm=True):
        """
        Initialize the GNN model.
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate for regularization
            gnn_type: Type of GNN layer ('gcn', 'sage', or 'gat')
            residual: Whether to use residual connections
            batch_norm: Whether to use batch normalization
        """
        super(YelpGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.residual = residual
        self.batch_norm = batch_norm
        
        # Layer lists for flexible architecture
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        
        # Input layer
        self.convs.append(self._create_conv_layer(input_dim, hidden_dim))
        if batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for i in range(num_layers - 2):
            self.convs.append(self._create_conv_layer(hidden_dim, hidden_dim))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(self._create_conv_layer(hidden_dim, output_dim))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(output_dim))
    
    def _create_conv_layer(self, in_dim, out_dim):
        """Create a specific type of GNN convolution layer."""
        if self.gnn_type == 'gcn':
            return GCNConv(in_dim, out_dim)
        elif self.gnn_type == 'sage':
            return SAGEConv(in_dim, out_dim)
        elif self.gnn_type == 'gat':
            return GATConv(in_dim, out_dim, heads=1)
        else:
            raise ValueError(f"Unknown GNN type: {self.gnn_type}")
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass through the network.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity in COO format [2, num_edges]
            edge_weight: Edge weights [num_edges]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        prev_x = None  # For residual connections
        
        for i, conv in enumerate(self.convs):
            # Save previous layer output for residual connections
            if self.residual and i > 0:
                prev_x = x
            
            # GNN layer
            if edge_weight is not None:
                x = conv(x, edge_index, edge_weight)
            else:
                x = conv(x, edge_index)
            
            # Apply batch norm if enabled
            if self.batch_norm and i < len(self.convs) - 1:  # No batch norm on last layer
                x = self.batch_norms[i](x)
            
            # Add residual connection if enabled
            if self.residual and i > 0 and prev_x is not None and x.size(-1) == prev_x.size(-1):
                x = x + prev_x
            
            # Apply activation and dropout (except last layer)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class YelpRecommender(nn.Module):
    """
    End-to-end recommendation model combining GNN with final prediction layers.
    """
    def __init__(self, gnn_model, prediction_dim=16):
        super(YelpRecommender, self).__init__()
        self.gnn = gnn_model
        self.embedding_dim = gnn_model.output_dim
        
        # Final prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, prediction_dim),
            nn.ReLU(),
            nn.Linear(prediction_dim, 1)
        )
    
    def forward(self, x, edge_index, edge_weight=None, user_indices=None, business_indices=None):
        """
        Forward pass through the full recommendation model.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            edge_weight: Edge weights (optional)
            user_indices: Indices of users in the recommendation pairs
            business_indices: Indices of businesses in the recommendation pairs
            
        Returns:
            Predicted scores for user-business pairs
        """
        # Get node embeddings from GNN
        embeddings = self.gnn(x, edge_index, edge_weight)
        
        # If we're in inference mode without specific pairs
        if user_indices is None or business_indices is None:
            return embeddings
        
        # Get user and business embeddings
        user_embeds = embeddings[user_indices]
        business_embeds = embeddings[business_indices]
        
        # Concatenate user and business embeddings
        pair_embeds = torch.cat([user_embeds, business_embeds], dim=1)
        
        # Predict rating
        scores = self.predictor(pair_embeds).squeeze()
        
        return scores