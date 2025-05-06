import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from data import Neo4jDataLoader
from model import YelpGNN, YelpRecommender
from utils import YelpTrainer

# Neo4j connection configuration
NEO4J_URI = "bolt://localhost:7687"  # Update if your Neo4j instance is hosted elsewhere
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # Change this to your actual password

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Yelp GNN Recommendation System')
    
    # Data loading parameters
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--cache_dir', type=str, default='cache', help='Directory to cache loaded data')
    parser.add_argument('--use_cached', action='store_true', help='Use cached data if available')
    
    # Model parameters
    parser.add_argument('--input_dim', type=int, default=64, help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=32, help='Output dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--gnn_type', type=str, default='sage', choices=['gcn', 'sage', 'gat'], help='GNN type')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--grad_accum_steps', type=int, default=2, help='Gradient accumulation steps')
    
    # Neighbor sampling parameters
    parser.add_argument('--num_neighbors', type=str, default='15,10', help='Number of neighbors to sample at each layer (comma-separated)')
    
    # Misc parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode: train or test')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path for testing')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def plot_training_history(history, save_path):
    """Plot training history and save to file."""
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    
    # Plot metrics
    plt.subplot(1, 2, 2)
    plt.plot(history['val_rmse'], label='RMSE')
    plt.plot(history['val_mae'], label='MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.title('Validation Metrics')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    """Main function."""
    args = parse_args()
    set_seed(args.seed)
    
    # Parse list arguments
    args.num_neighbors = [int(n) for n in args.num_neighbors.split(',')]
    
    # Create cache directory if needed
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
    
    # Initialize data loader
    data_loader = Neo4jDataLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, batch_size=args.batch_size)
    
    try:
        # Load graph data
        cache_dir = args.cache_dir if args.use_cached else None
        graph_data = data_loader.load_graph_data(cache_dir=cache_dir)
        
        # Create train/test split
        train_data, val_data, test_data = data_loader.create_train_test_split(cache_dir=cache_dir)
        
        print(f"Graph data loaded: {graph_data}")
        print(f"Number of nodes: {graph_data.num_nodes}")
        print(f"Number of edges: {graph_data.edge_index.size(1)}")
        print(f"Feature dimension: {graph_data.x.size(1)}")
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Test samples: {len(test_data)}")
        
        # Initialize model
        input_dim = graph_data.x.size(1)
        gnn_model = YelpGNN(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            gnn_type=args.gnn_type,
            residual=True,
            batch_norm=True
        )
        
        recommender_model = YelpRecommender(gnn_model)
        
        # Initialize trainer
        trainer = YelpTrainer(recommender_model, device=args.device)
        
        if args.mode == 'train':
            # Train model
            history = trainer.train(
                graph_data=graph_data,
                train_data=train_data,
                val_data=val_data,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                gradient_accumulation_steps=args.grad_accum_steps,
                num_neighbors=args.num_neighbors,
                checkpoint_dir=args.checkpoint_dir
            )
            
            # Plot training history
            plot_training_history(history, os.path.join(args.checkpoint_dir, 'training_history.png'))
            
            # Load best model and evaluate on test set
            trainer.load_checkpoint(os.path.join(args.checkpoint_dir, 'best_model.pt'))
        else:
            # Load model from checkpoint
            if args.checkpoint:
                trainer.load_checkpoint(args.checkpoint)
            else:
                trainer.load_checkpoint(os.path.join(args.checkpoint_dir, 'best_model.pt'))
        
        # Evaluate on test set
        predictions, targets = trainer.predict(
            graph_data=graph_data,
            test_data=test_data,
            batch_size=args.batch_size,
            num_neighbors=args.num_neighbors
        )
        
    finally:
        # Close data loader
        data_loader.close()

if __name__ == '__main__':
    main()