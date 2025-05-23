import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import subgraph
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score

class YelpTrainer:
    """
    Trainer class for Yelp GNN recommendation model.
    Implements memory-efficient training with gradient accumulation and early stopping.
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.train_loader = None
        self.val_loader = None
        print(f"Using device: {device}")
        
    def train(self, graph_data, train_data, val_data, 
              epochs=50, batch_size=1024, learning_rate=0.001, 
              weight_decay=1e-5, patience=10, gradient_accumulation_steps=1,
              num_neighbors=[15, 10], checkpoint_dir='checkpoints'):
        """
        Train the model.
        
        Args:
            graph_data: PyTorch Geometric Data object containing the graph
            train_data: Training data tensor
            val_data: Validation data tensor
            epochs: Number of epochs to train for
            batch_size: Batch size for training
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            patience: Patience for early stopping
            gradient_accumulation_steps: Number of steps to accumulate gradients
            num_neighbors: Number of neighbors to sample for each node
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Dict of training history
        """
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        
        graph_data = graph_data.to(self.device)
        x, edge_index, edge_attr = graph_data.x, graph_data.edge_index, graph_data.edge_attr
        
        
        train_data = train_data.to(self.device)
        val_data = val_data.to(self.device)
        
        
        self._setup_neighbor_sampler(graph_data, train_data, val_data, batch_size, num_neighbors)
        
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [],
            'val_mae': []
        }
        
        
        best_val_loss = float('inf')
        best_epoch = 0
        no_improve_epochs = 0
        
        
        for epoch in range(epochs):
            start_time = time.time()
            
            
            self.model.train()
            train_loss = self._train_epoch(
                optimizer, criterion, gradient_accumulation_steps
            )
            
            
            self.model.eval()
            val_metrics = self._validate(criterion)
            
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_rmse'].append(val_metrics['rmse'])
            history['val_mae'].append(val_metrics['mae'])
            
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_metrics['loss']:.4f} - "
                  f"Val RMSE: {val_metrics['rmse']:.4f} - "
                  f"Val MAE: {val_metrics['mae']:.4f}")
            
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_epoch = epoch
                no_improve_epochs = 0
                
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss']
                }, os.path.join(checkpoint_dir, 'best_model.pt'))
                
                print(f"Model improved, saved checkpoint at epoch {epoch+1}")
            else:
                no_improve_epochs += 1
                
            
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch+1}")
                break
            
            
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss']
                }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
        return history
    
    def _setup_neighbor_sampler(self, graph_data, train_data, val_data, batch_size, num_neighbors):
        """
        Set up neighbor sampling loaders for memory-efficient training.
        
        Args:
            graph_data: PyTorch Geometric Data object
            train_data: Training data tensor
            val_data: Validation data tensor
            batch_size: Batch size
            num_neighbors: Number of neighbors to sample at each layer
        """
        
        train_users = train_data[:, 0].unique()
        train_businesses = train_data[:, 1].unique()
        val_users = val_data[:, 0].unique()
        val_businesses = val_data[:, 1].unique()
        
        
        train_nodes = torch.cat([train_users, train_businesses]).unique()
        val_nodes = torch.cat([val_users, val_businesses]).unique()
        
        
        self.train_loader = NeighborLoader(
            graph_data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=train_nodes,
            shuffle=True
        )
        
        self.val_loader = NeighborLoader(
            graph_data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=val_nodes,
            shuffle=False
        )
        
        
        self.train_pairs = train_data
        self.val_pairs = val_data
    
    def _train_epoch(self, optimizer, criterion, gradient_accumulation_steps):
        """
        Train for one epoch.
        
        Args:
            optimizer: Optimizer
            criterion: Loss function
            gradient_accumulation_steps: Number of steps to accumulate gradients
            
        Returns:
            Average training loss
        """
        total_loss = 0
        num_batches = 0
        optimizer.zero_grad()
        
        
        for i, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            
            batch = batch.to(self.device)
            
            
            batch_users = self.train_pairs[:, 0].to(self.device)
            batch_businesses = self.train_pairs[:, 1].to(self.device)
            batch_n_id = batch.n_id.to(self.device)
            
            
            user_mask = torch.isin(batch_users, batch_n_id)
            business_mask = torch.isin(batch_businesses, batch_n_id)
            pair_mask = user_mask & business_mask
            
            if not pair_mask.any():
                continue  
            
            batch_pairs = self.train_pairs[pair_mask]
            
            
            batch_user_indices = []
            batch_business_indices = []
            
            for user_idx in batch_pairs[:, 0]:
                match_indices = torch.where(batch_n_id == user_idx)[0]
                if len(match_indices) > 0:
                    batch_user_indices.append(match_indices[0].item())
                    
            for business_idx in batch_pairs[:, 1]:
                match_indices = torch.where(batch_n_id == business_idx)[0]
                if len(match_indices) > 0:
                    batch_business_indices.append(match_indices[0].item())
            
            
            batch_user_indices = torch.tensor(batch_user_indices, device=self.device)
            batch_business_indices = torch.tensor(batch_business_indices, device=self.device)
            batch_stars = batch_pairs[:, 2].float().to(self.device)
            
            
            min_len = min(len(batch_user_indices), len(batch_business_indices), len(batch_stars))
            if min_len == 0:
                continue
                
            batch_user_indices = batch_user_indices[:min_len]
            batch_business_indices = batch_business_indices[:min_len]
            batch_stars = batch_stars[:min_len]
            
            
            scores = self.model(batch.x, batch.edge_index, 
                              user_indices=batch_user_indices, 
                              business_indices=batch_business_indices)
            
            
            loss = criterion(scores, batch_stars)
            
            
            loss = loss / gradient_accumulation_steps
            
            
            loss.backward()
            
            
            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            
            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
        
        return total_loss / max(1, num_batches)
    
    def _validate(self, criterion):
        """
        Validate the model.
        
        Args:
            criterion: Loss function
            
        Returns:
            Dict of validation metrics
        """
        total_loss = 0
        all_preds = []
        all_targets = []
        num_batches = 0
        
        with torch.no_grad():
            
            for batch in tqdm(self.val_loader, desc="Validating"):
                
                batch = batch.to(self.device)
                
                
                batch_users = self.val_pairs[:, 0].to(self.device)
                batch_businesses = self.val_pairs[:, 1].to(self.device)
                batch_n_id = batch.n_id.to(self.device)
                
                
                user_mask = torch.isin(batch_users, batch_n_id)
                business_mask = torch.isin(batch_businesses, batch_n_id)
                pair_mask = user_mask & business_mask
                
                if not pair_mask.any():
                    continue  
                
                batch_pairs = self.val_pairs[pair_mask]
                
                
                batch_user_indices = []
                batch_business_indices = []
                
                for user_idx in batch_pairs[:, 0]:
                    match_indices = torch.where(batch_n_id == user_idx)[0]
                    if len(match_indices) > 0:
                        batch_user_indices.append(match_indices[0].item())
                        
                for business_idx in batch_pairs[:, 1]:
                    match_indices = torch.where(batch_n_id == business_idx)[0]
                    if len(match_indices) > 0:
                        batch_business_indices.append(match_indices[0].item())
                
                
                batch_user_indices = torch.tensor(batch_user_indices, device=self.device)
                batch_business_indices = torch.tensor(batch_business_indices, device=self.device)
                batch_stars = batch_pairs[:, 2].float().to(self.device)
                
                
                min_len = min(len(batch_user_indices), len(batch_business_indices), len(batch_stars))
                if min_len == 0:
                    continue
                    
                batch_user_indices = batch_user_indices[:min_len]
                batch_business_indices = batch_business_indices[:min_len]
                batch_stars = batch_stars[:min_len]
                
                
                scores = self.model(batch.x, batch.edge_index, 
                                  user_indices=batch_user_indices, 
                                  business_indices=batch_business_indices)
                
                
                loss = criterion(scores, batch_stars)
                
                
                total_loss += loss.item()
                all_preds.extend(scores.cpu().numpy())
                all_targets.extend(batch_stars.cpu().numpy())
                num_batches += 1
        
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        mae = mean_absolute_error(all_targets, all_preds)
        
        return {
            'loss': total_loss / max(1, num_batches),
            'rmse': rmse,
            'mae': mae
        }
    
    def predict(self, graph_data, test_data, batch_size=1024, num_neighbors=[15, 10]):
        """
        Make predictions on test data.
        
        Args:
            graph_data: PyTorch Geometric Data object
            test_data: Test data tensor
            batch_size: Batch size
            num_neighbors: Number of neighbors to sample at each layer
            
        Returns:
            Numpy array of predictions
        """
        self.model.eval()
        graph_data = graph_data.to(self.device)
        test_data = test_data.to(self.device)
        
        
        test_users = test_data[:, 0].unique()
        test_businesses = test_data[:, 1].unique()
        test_nodes = torch.cat([test_users, test_businesses]).unique()
        
        test_loader = NeighborLoader(
            graph_data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=test_nodes,
            shuffle=False
        )
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            
            for batch in tqdm(test_loader, desc="Testing"):
                
                batch = batch.to(self.device)
                
                
                batch_users = test_data[:, 0].to(self.device)
                batch_businesses = test_data[:, 1].to(self.device)
                batch_n_id = batch.n_id.to(self.device)
                
                
                user_mask = torch.isin(batch_users, batch_n_id)
                business_mask = torch.isin(batch_businesses, batch_n_id)
                pair_mask = user_mask & business_mask
                
                if not pair_mask.any():
                    continue  
                
                batch_pairs = test_data[pair_mask]
                
                
                batch_user_indices = []
                batch_business_indices = []
                
                for user_idx in batch_pairs[:, 0]:
                    match_indices = torch.where(batch_n_id == user_idx)[0]
                    if len(match_indices) > 0:
                        batch_user_indices.append(match_indices[0].item())
                        
                for business_idx in batch_pairs[:, 1]:
                    match_indices = torch.where(batch_n_id == business_idx)[0]
                    if len(match_indices) > 0:
                        batch_business_indices.append(match_indices[0].item())
                
                
                batch_user_indices = torch.tensor(batch_user_indices, device=self.device)
                batch_business_indices = torch.tensor(batch_business_indices, device=self.device)
                batch_stars = batch_pairs[:, 2].float().to(self.device)
                
                
                min_len = min(len(batch_user_indices), len(batch_business_indices), len(batch_stars))
                if min_len == 0:
                    continue
                    
                batch_user_indices = batch_user_indices[:min_len]
                batch_business_indices = batch_business_indices[:min_len]
                batch_stars = batch_stars[:min_len]
                
                
                scores = self.model(batch.x, batch.edge_index, 
                                  user_indices=batch_user_indices, 
                                  business_indices=batch_business_indices)
                
                
                all_preds.extend(scores.cpu().numpy())
                all_targets.extend(batch_stars.cpu().numpy())
        
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        mae = mean_absolute_error(all_targets, all_preds)
        
        print(f"Test RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        return all_preds, all_targets
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from checkpoint (epoch {checkpoint['epoch']+1})")
        
        return checkpoint