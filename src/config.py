"""
Configuration parameters for the IMDB Review Sentiment Classification project.
"""
import os
import torch

class Config:
    # Project paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    CACHE_DIR = os.path.join(BASE_DIR, "dataset")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
    FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
    METRICS_DIR = os.path.join(OUTPUTS_DIR, "metrics")
    SCREENSHOTS_DIR = os.path.join(OUTPUTS_DIR, "screenshots")
    
    # Dataset and split
    RANDOM_SEED = 42
    VAL_SIZE = 0.20  # 20% validation split from training pool
    
    # Preprocessing
    VOCAB_SIZE = 30000
    MIN_FREQ = 2
    MAX_SEQ_LEN = 256
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    PAD_IDX = 0
    UNK_IDX = 1
    
    # BiLSTM Architecture
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 128
    NUM_LAYERS = 1
    BIDIRECTIONAL = True
    DROPOUT = 0.3
    DENSE_DIM = 64
    
    # Training
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 8
    EARLY_STOPPING_PATIENCE = 3
    LR_PATIENCE = 1
    LR_FACTOR = 0.5
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
