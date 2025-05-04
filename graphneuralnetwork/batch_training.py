import torch
import numpy as np
from torch_geometric.data import NeighborSampler
from collections import defaultdict


class EfficientNeighborSampler:
    """Memory-efficient neighborhood sampling for large graphs"""
    
    def __init__(self, edge_index, num_nodes, batch_size=4096, sizes=[15, 10], 
                 num_workers=4, shuffle=True):
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.sizes = sizes
        self.num_workers = num_workers
        self.shuffle = shuffle
        
        # Create adjacency list for efficient sampling
        self._create_adj_list()
        
    def _create_adj_list(self):
        """Create adjacency list from edge index"""
        self.adj_list = defaultdict(list)
        
        # Convert edge index to adjacency list
        edge_index_numpy = self.edge_index.cpu().numpy()
        for i in range(edge_index_numpy.shape[1]):
            src, dst = edge_index_numpy[0, i], edge_index_numpy[1, i]
            self.adj_list[src].append(dst)
            
    def sample_neighbors(self, batch_nodes, sizes):
        """Sample neighbors for a batch of nodes"""
        output_nodes = batch_nodes.tolist()
        adjs = []
        
        # For each hop, sample neighbors
        for size in sizes:
            # Initialize lists for edges
            edge_index_src = []
            edge_index_dst = []
            
            # For each node in the current batch, sample its neighbors
            for node in output_nodes:
                neighbors = self.adj_list.get(node, [])
                
                # If there are more neighbors than the sampling size, randomly sample
                if len(neighbors) > size:
                    sampled_neighbors = np.random.choice(neighbors, size, replace=False).tolist()
                else:
                    sampled_neighbors = neighbors
                
                # Add edges
                for neighbor in sampled_neighbors:
                    edge_index_src.append(node)
                    edge_index_dst.append(neighbor)
            
            # Create edge index for current hop
            if edge_index_src:
                hop_edge_index = torch.tensor([edge_index_src, edge_index_dst], dtype=torch.long)
                adjs.append((hop_edge_index, None, None))  # Format compatible with PyG's sampler
            else:
                # No edges for this hop
                adjs.append((torch.empty((2, 0), dtype=torch.long), None, None))
            
            # Update output nodes for next hop
            if sizes[-1] != -1:  # If we're not sampling the entire neighborhood in the last hop
                output_nodes = list(set(edge_index_dst))
        
        return adjs, output_nodes
    
    def __iter__(self):
        """Create batches of nodes for mini-batch training"""
        node_idx = list(range(self.num_nodes))
        
        if self.shuffle:
            np.random.shuffle(node_idx)
            
        # Create batches
        for batch_start in range(0, len(node_idx), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(node_idx))
            batch = torch.tensor(node_idx[batch_start:batch_end], dtype=torch.long)
            
            # Sample neighbors for current batch
            adjs, output_nodes = self.sample_neighbors(batch, self.sizes)
            
            yield batch, output_nodes, adjs


class ClusteredSampler:
    """Clustered sampling for improved training efficiency"""
    
    def __init__(self, edge_index, num_nodes, num_clusters=100, batch_size=1024, 
                 sizes=[15, 10], num_workers=4):
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        self.batch_size = batch_size
        self.sizes = sizes
        self.num_workers = num_workers
        
        # Assign nodes to clusters (simple random assignment for demo)
        self.clusters = self._create_clusters()
        
        # Create efficient neighbor sampler
        self.neighbor_sampler = EfficientNeighborSampler(
            edge_index, num_nodes, batch_size, sizes, num_workers
        )
        
    def _create_clusters(self):
        """Assign nodes to clusters randomly"""
        clusters = [[] for _ in range(self.num_clusters)]
        for node in range(self.num_nodes):
            cluster_id = node % self.num_clusters
            clusters[cluster_id].append(node)
        return clusters
    
    def __iter__(self):
        """Iterate through clusters"""
        # Shuffle cluster order
        cluster_order = np.random.permutation(self.num_clusters)
        
        for cluster_id in cluster_order:
            nodes = self.clusters[cluster_id]
            
            # Skip empty clusters
            if not nodes:
                continue
                
            # Create batches from current cluster
            np.random.shuffle(nodes)
            
            for batch_start in range(0, len(nodes), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(nodes))
                batch = torch.tensor(nodes[batch_start:batch_end], dtype=torch.long)
                
                # Sample neighbors for current batch
                adjs, output_nodes = self.neighbor_sampler.sample_neighbors(batch, self.sizes)
                
                yield batch, output_nodes, adjs


def create_memory_efficient_loader(edge_index, num_nodes, batch_size=4096, 
                                  sizes=[15, 10], num_workers=4, method='neighborhood'):
    """Create an appropriate sampler based on graph size and available memory"""
    if method == 'neighborhood':
        return EfficientNeighborSampler(
            edge_index, num_nodes, batch_size, sizes, num_workers
        )
    elif method == 'clustered':
        return ClusteredSampler(
            edge_index, num_nodes, num_clusters=100, batch_size=batch_size, 
            sizes=sizes, num_workers=num_workers
        )
    else:
        # Fallback to PyTorch Geometric's sampler
        return NeighborSampler(
            edge_index, node_idx=None, sizes=sizes, batch_size=batch_size,
            shuffle=True, num_workers=num_workers
        )
