# Neo4j Database Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"

# Training Configuration
DEVICE = "cuda"  # or "cpu" if no GPU available
BATCH_SIZE = 2048
EPOCHS = 30
LEARNING_RATE = 0.001
HIDDEN_DIMS = [64, 128, 64, 32]
DROPOUT = 0.3
WEIGHT_DECAY = 1e-5

# Data Processing Configuration
CACHE_DIR = "./cache"
TEST_RATIO = 0.2