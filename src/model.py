"""
Neural Network Model architectures for IMDb Review Sentiment Classification.
Includes the primary Bidirectional LSTM (BiLSTM) network.
"""
import torch
import torch.nn as nn
from src.config import Config

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional Long Short-Term Memory (BiLSTM) Network for binary sentiment classification.
    
    Architecture:
    1. Embedding layer (vocab_size -> embedding_dim) with padding index
    2. Spatial / Embedding Dropout
    3. Bidirectional LSTM layer (hidden_dim in each direction)
    4. Global Max Pooling over time (capturing salient sentiment indicators across sequence)
    5. Fully connected dense layer with ReLU activation
    6. Dropout for regularization
    7. Linear projection to single scalar logit (compatible with BCEWithLogitsLoss)
    """
    def __init__(
        self,
        vocab_size: int = Config.VOCAB_SIZE,
        embedding_dim: int = Config.EMBEDDING_DIM,
        hidden_dim: int = Config.HIDDEN_DIM,
        num_layers: int = Config.NUM_LAYERS,
        dense_dim: int = Config.DENSE_DIM,
        dropout: float = Config.DROPOUT,
        pad_idx: int = Config.PAD_IDX
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding_dropout = nn.Dropout(dropout)
        
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # BiLSTM outputs hidden_dim in forward direction and hidden_dim in backward direction = 2 * hidden_dim
        lstm_out_dim = hidden_dim * 2
        
        # Dense hidden projection layer
        self.fc1 = nn.Linear(lstm_out_dim, dense_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Output logit unit
        self.fc_out = nn.Linear(dense_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.embedding_dropout(embedded)
        
        lstm_out, (h_n, c_n) = self.bilstm(embedded)  # (batch_size, seq_len, 2 * hidden_dim)
        
        # Global max pooling over sequence length to extract most prominent features
        pooled, _ = torch.max(lstm_out, dim=1)  # (batch_size, 2 * hidden_dim)
        
        # Dense hidden layer
        dense = self.fc1(pooled)
        activated = self.relu(dense)
        dropped = self.dropout(activated)
        
        # Output logit
        logits = self.fc_out(dropped).squeeze(1)  # (batch_size,)
        return logits

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = BiLSTMClassifier(vocab_size=30000, embedding_dim=128, hidden_dim=128)
    dummy_input = torch.randint(0, 1000, (4, 256))
    output = model(dummy_input)
    print("Dummy output logits shape:", output.shape)
    print("Trainable parameters:", model.count_parameters())
