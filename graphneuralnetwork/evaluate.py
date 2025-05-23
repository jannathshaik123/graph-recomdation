import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score
from data import Neo4jDataLoader
from model import YelpGNN, YelpRecommender
from utils import YelpTrainer


NEO4J_URI = "bolt://localhost:7687"  
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Yelp GNN Recommendation System')
    
    
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for evaluation')
    parser.add_argument('--cache_dir', type=str, default='cache', help='Directory with cached data')
    
    
    parser.add_argument('--input_dim', type=int, default=64, help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=32, help='Output dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--gnn_type', type=str, default='sage', choices=['gcn', 'sage', 'gat'], help='GNN type')
    
    
    parser.add_argument('--num_neighbors', type=str, default='15,10', help='Number of neighbors to sample at each layer (comma-separated)')
    
    
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory with saved checkpoints')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt', help='Checkpoint filename')
    
    
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save evaluation results')
    parser.add_argument('--threshold', type=float, default=3.0, help='Threshold for classification metrics')
    
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    return parser.parse_args()

def calculate_metrics(predictions, targets, threshold=3.0):
    """
    Calculate evaluation metrics for regression and classification.
    
    Args:
        predictions: Model predictions (numpy array)
        targets: Ground truth values (numpy array)
        threshold: Threshold for binarizing ratings into positive/negative
        
    Returns:
        Dict containing all the metrics
    """
    
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    
    binary_preds = (predictions >= threshold).astype(int)
    binary_targets = (targets >= threshold).astype(int)
    
    precision = precision_score(binary_targets, binary_preds, zero_division=0)
    recall = recall_score(binary_targets, binary_preds, zero_division=0)
    f1 = f1_score(binary_targets, binary_preds, zero_division=0)
    
    
    pred_dist = np.histogram(predictions, bins=5, range=(1, 6))[0]
    target_dist = np.histogram(targets, bins=5, range=(1, 6))[0]
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pred_distribution': pred_dist,
        'target_distribution': target_dist
    }

def plot_metrics(metrics, output_path):
    """
    Plot evaluation metrics and save the figures.
    
    Args:
        metrics: Dict containing metrics
        output_path: Path to save the plots
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    
    plt.figure(figsize=(10, 6))
    plt.bar(['RMSE', 'MAE'], [metrics['rmse'], metrics['mae']])
    plt.title('Regression Metrics')
    plt.ylabel('Error')
    plt.savefig(os.path.join(output_path, 'regression_metrics.png'))
    plt.close()
    
    
    plt.figure(figsize=(10, 6))
    plt.bar(['Precision', 'Recall', 'F1'], [metrics['precision'], metrics['recall'], metrics['f1']])
    plt.title('Classification Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_path, 'classification_metrics.png'))
    plt.close()
    
    
    plt.figure(figsize=(12, 6))
    
    
    bar_width = 0.35
    positions1 = np.arange(5)
    positions2 = positions1 + bar_width
    
    
    plt.bar(positions1, metrics['target_distribution'], bar_width, label='Actual Ratings')
    plt.bar(positions2, metrics['pred_distribution'], bar_width, label='Predicted Ratings')
    
    
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title('Rating Distribution')
    plt.xticks(positions1 + bar_width / 2, ['1', '2', '3', '4', '5'])
    plt.legend()
    
    plt.savefig(os.path.join(output_path, 'rating_distribution.png'))
    plt.close()

def main():
    
    args = parse_args()
    
    
    num_neighbors = [int(n) for n in args.num_neighbors.split(',')]
    
    
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    
    data_loader = Neo4jDataLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, batch_size=args.batch_size)
    
    try:
        
        print("Loading graph data from cache...")
        graph_data = data_loader.load_graph_data(cache_dir=args.cache_dir)
        
        
        _, _, test_data = data_loader.create_train_test_split(cache_dir=args.cache_dir)
        
        print(f"Graph data loaded: {graph_data}")
        print(f"Number of nodes: {graph_data.num_nodes}")
        print(f"Number of edges: {graph_data.edge_index.size(1)}")
        print(f"Feature dimension: {graph_data.x.size(1)}")
        print(f"Test samples: {len(test_data)}")
        
        
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
        
        
        trainer = YelpTrainer(recommender_model, device=args.device)
        
        
        trainer.load_checkpoint(checkpoint_path)
        
        
        print("Evaluating model on test data...")
        predictions, targets = trainer.predict(
            graph_data=graph_data,
            test_data=test_data,
            batch_size=args.batch_size,
            num_neighbors=num_neighbors
        )
        
        
        metrics = calculate_metrics(predictions, targets, threshold=args.threshold)
        
        
        metrics_df = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'R²', 'Precision', 'Recall', 'F1'],
            'Value': [metrics['rmse'], metrics['mae'], metrics['r2'], 
                     metrics['precision'], metrics['recall'], metrics['f1']]
        })
        metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
        
        
        with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
            f.write("Yelp Recommendation System Evaluation Report\n")
            f.write("=========================================\n\n")
            f.write(f"Model: {args.gnn_type.upper()} GNN with {args.num_layers} layers\n")
            f.write(f"Checkpoint: {args.checkpoint}\n\n")
            
            f.write("Regression Metrics:\n")
            f.write(f"- RMSE: {metrics['rmse']:.4f}\n")
            f.write(f"- MAE: {metrics['mae']:.4f}\n")
            f.write(f"- R²: {metrics['r2']:.4f}\n\n")
            
            f.write("Classification Metrics (threshold = {args.threshold}):\n")
            f.write(f"- Precision: {metrics['precision']:.4f}\n")
            f.write(f"- Recall: {metrics['recall']:.4f}\n")
            f.write(f"- F1 Score: {metrics['f1']:.4f}\n\n")
            
            f.write("Rating Distribution:\n")
            for i in range(5):
                f.write(f"- Rating {i+1}: {metrics['target_distribution'][i]} actual, {metrics['pred_distribution'][i]} predicted\n")
        
        
        plot_metrics(metrics, output_dir)
        
        print(f"Evaluation complete. Results saved to {output_dir}")
    
    finally:
        
        data_loader.close()

if __name__ == '__main__':
    main()