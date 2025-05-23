# Yelp Business Recommendation System using Knowledge Graphs

A comprehensive recommendation system that leverages graph databases and neural networks to provide personalized business recommendations from Yelp data. This project implements three different recommendation approaches: collaborative filtering, graph-based recommendations, and Graph Neural Networks (GNNs).

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Performance Results](#performance-results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project was developed as part of CS F377 Design Project at BITS Pilani Dubai Campus. It addresses the challenge of providing accurate business recommendations by combining traditional collaborative filtering with modern graph-based approaches and deep learning techniques.

### Key Problems Solved:

- **Data Sparsity**: Most users review only a small fraction of businesses
- **Cold-Start Problem**: Difficulty recommending to new users or businesses
- **Complex Relationships**: Capturing multi-dimensional relationships between users, businesses, and attributes
- **Scalability**: Handling large-scale real-world data efficiently

## ✨ Features

- **Multiple Recommendation Strategies**:

  - Collaborative Filtering (User-based, Item-based, Matrix Factorization)
  - Graph-based Recommendations using Neo4j
  - Graph Neural Networks (GCN, GraphSAGE, GAT)

- **Interactive Web Interface**: Built with Streamlit for easy exploration

- **Comprehensive Evaluation**: Multiple metrics including RMSE, MAE, Precision@4, Recall@4

- **Real-time Performance**: Optimized for interactive use

- **Explainable Recommendations**: Clear reasoning behind suggestions

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Processing     │    │   Recommendation│
│                 │    │     Layer       │    │     Engines     │
│ • Yelp Dataset  │───▶│ • Feature Eng.  │───▶│ • Collaborative │
│ • JSON Files    │    │ • Preprocessing │    │ • Graph-based   │
│ • CSV Exports   │    │ • Visualization │    │ • GNN Models    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐             │
│  Storage Layer  │    │   Interface     │             │
│                 │    │     Layer       │             │
│ • Neo4j Graph   │◀───│ • Streamlit UI  │◀────────────┘
│ • Matrix Store  │    │ • Visualization │
│ • Model Cache   │    │ • User Controls │
└─────────────────┘    └─────────────────┘
```

## 🔧 Installation

### Prerequisites

- Python 3.8+
- Neo4j 4.4+
- 16GB RAM (recommended)
- CUDA-compatible GPU (optional, for GNN training)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/yelp-recommendation-system.git
cd yelp-recommendation-system
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Python Packages:

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
streamlit>=1.12.0
neo4j>=4.4.0
torch>=1.10.0
torch-geometric>=2.0.0
nltk>=3.6.0
scipy>=1.7.0
```

### Step 4: Setup Neo4j

1. Download and install Neo4j Desktop
2. Create a new database with:
   - Bolt port: 7687
   - Username: neo4j
   - Password: password (or update in config files)
3. Allocate 8GB heap memory in neo4j.conf

## 📊 Dataset Setup

### Step 1: Download Yelp Dataset

1. Download the Yelp Dataset from the official Yelp Dataset Challenge
2. Extract files to `data/yelp_training_set/` directory:
   ```
   data/
   └── yelp_training_set/
       ├── yelp_training_set_business.json
       ├── yelp_training_set_review.json
       ├── yelp_training_set_user.json
       └── yelp_training_set_checkin.json
   ```

### Step 2: Extract and Preprocess Data

```bash
python extract.py
```

This creates CSV files in the `extract/` directory with cleaned business and user data.

### Step 3: Load Data into Neo4j

```bash
python database.py
```

This populates the Neo4j database with:

- Business nodes with categories and locations
- User nodes with profiles
- Review relationships
- Check-in patterns

## 🚀 Usage

### Option 1: Interactive Streamlit Interface

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` to access the web interface.

### Option 2: Collaborative Filtering

```bash
python utils.py
```

Runs collaborative filtering with matrix factorization and evaluates performance.

### Option 3: Graph Neural Network Training

```bash
python main.py --mode train --epochs 50 --batch_size 1024
```

### Option 4: Graph-based Recommendations

```bash
python utils.py  # (the Neo4j-based recommendation system)
```

## 📁 File Structure

```
yelp-recommendation-system/
│
├── extract.py              # Data extraction and preprocessing
├── utils.py                # Collaborative filtering implementation
├── data.py                 # Neo4j data loader for GNN
├── main.py                 # GNN training and evaluation
├── model.py                # GNN model architectures
├── database.py             # Neo4j database setup
├── utils.py                # Graph-based recommendation system
│
├── data/                   # Dataset directory
│   └── yelp_training_set/  # Raw Yelp JSON files
│
├── extract/                # Processed CSV files
│   ├── business_data.csv
│   └── user_data.csv
│
├── models/                 # Trained model storage
├── cache/                  # Cached data for faster loading
├── checkpoints/            # GNN model checkpoints
│
└── requirements.txt        # Python dependencies
```

### Key Files Explained:

#### `extract.py`

- Loads large JSON files efficiently
- Extracts business and user data
- Creates clean CSV exports
- Handles missing values and data types

#### `utils.py` (Collaborative Filtering)

- Implements memory-efficient collaborative filtering
- User-based and item-based similarity
- Matrix factorization with SGD
- Bounded predictions (1-5 star range)
- Evaluation metrics and visualization

#### `database.py`

- Creates Neo4j knowledge graph
- Establishes relationships between entities
- Optimizes with constraints and indexes
- Batch processing for large datasets

#### `data.py`

- PyTorch Geometric data loader
- Converts Neo4j data to graph tensors
- Memory-efficient neighbor sampling
- Train/validation/test splits

#### `model.py`

- GNN architectures (GCN, GraphSAGE, GAT)
- Residual connections and batch normalization
- Recommendation-specific design
- Rating prediction layers

#### `main.py`

- GNN training pipeline
- Hyperparameter optimization
- Model evaluation and checkpointing
- Command-line interface

## 📈 Performance Results

### Model Comparison

| Method                          | RMSE      | MAE       | Precision@4 | Recall@4  | F1@4      |
| ------------------------------- | --------- | --------- | ----------- | --------- | --------- |
| Baseline                        | 1.02      | 0.794     | 0.843       | 0.338     | 0.483     |
| User-based CF                   | 1.183     | 0.902     | 0.768       | 0.474     | 0.586     |
| Item-based CF                   | 1.157     | 0.894     | 0.792       | 0.491     | 0.606     |
| Hybrid CF                       | 1.101     | 0.849     | 0.802       | 0.454     | 0.58      |
| Matrix Factorization (Graph DB) | **0.843** | **0.661** | -           | -         | -         |
| SAGE GNN                        | 1.898     | 1.592     | 0.674       | **0.874** | **0.761** |

### Key Insights:

- **Matrix Factorization** achieves best rating prediction accuracy
- **SAGE GNN** excels at generating comprehensive recommendation lists
- **22.5% improvement** in cold-start scenarios using graph-based methods
- **Item-based CF** outperforms user-based for this dataset

## 🎮 Usage Examples

### Generate Recommendations for a User

```python
from utils import YelpRecommendationSystem

# Initialize system
recommender = YelpRecommendationSystem("bolt://localhost:7687", "neo4j", "password")

# Get hybrid recommendations
recommendations = recommender.hybrid_recommendations("user_123", top_n=10)

for rec in recommendations:
    print(f"{rec['name']} - Predicted Rating: {rec['predicted_rating']:.2f}")
```

### Train Matrix Factorization Model

```python
# Train new model
user_factors, business_factors, global_avg = recommender.train_matrix_factorization(
    num_factors=20,
    learning_rate=0.005,
    num_iterations=50
)

# Save model
recommender.save_matrix_factorization_model(user_factors, business_factors, global_avg)
```

### Train GNN Model

```bash
python main.py \
    --mode train \
    --gnn_type sage \
    --hidden_dim 128 \
    --num_layers 3 \
    --epochs 100 \
    --batch_size 1024 \
    --lr 0.001
```

## 🔧 Configuration

### Neo4j Configuration

Update connection settings in the Python files:

```python
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"
```

### GNN Hyperparameters

Key parameters in `main.py`:

- `--hidden_dim`: Hidden layer dimensions (64, 128, 256)
- `--num_layers`: Number of GNN layers (2-4)
- `--dropout`: Dropout rate (0.1-0.5)
- `--gnn_type`: GNN architecture (gcn, sage, gat)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **BITS Pilani Dubai Campus** for academic support
- **Dr. Sujala D. Shetty** for project supervision

For questions or support, please open an issue in this repository.

---

**Note**: This project was developed as part of CS F377 Design Project coursework. Please ensure you have appropriate permissions for the Yelp dataset before use.
