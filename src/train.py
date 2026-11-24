"""
Training script for the Bidirectional LSTM model.
Includes reproducible random seeding, validation monitoring, early stopping,
learning rate scheduling, model checkpointing, and training history logging.
"""
import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from src.config import Config
from src.data_loader import get_train_val_test_splits
from src.preprocessing import Vocab, get_dataloader
from src.model import BiLSTMClassifier

def set_seed(seed=Config.RANDOM_SEED):
    """Ensure strict reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        
        # Clip gradients to avoid exploding gradients in LSTM
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
        
        preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(batch_y.cpu().numpy())
        
    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc

def evaluate_val(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_y.cpu().numpy())
            
    val_loss = total_loss / len(dataloader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)
    val_f1 = f1_score(all_targets, all_preds)
    return val_loss, val_acc, val_f1

def train_bilstm():
    print("=" * 60)
    print("STARTING BiLSTM MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    set_seed(Config.RANDOM_SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    # 1. Load data
    train_df, val_df, test_df = get_train_val_test_splits()
    print(f"Data splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)} (untouched)")
    
    # 2. Build vocabulary strictly from training subset
    vocab_path = os.path.join(Config.MODELS_DIR, "vocab.json")
    vocab = Vocab(
        pad_token=Config.PAD_TOKEN,
        unk_token=Config.UNK_TOKEN,
        pad_idx=Config.PAD_IDX,
        unk_idx=Config.UNK_IDX
    )
    vocab.build_vocab(train_df["text"].tolist(), max_size=Config.VOCAB_SIZE, min_freq=Config.MIN_FREQ)
    vocab.save(vocab_path)
    
    # 3. Create DataLoaders
    print("Creating DataLoaders...")
    train_loader = get_dataloader(
        train_df["text"].tolist(), train_df["label"].tolist(),
        vocab, batch_size=Config.BATCH_SIZE, shuffle=True, max_len=Config.MAX_SEQ_LEN
    )
    val_loader = get_dataloader(
        val_df["text"].tolist(), val_df["label"].tolist(),
        vocab, batch_size=Config.BATCH_SIZE, shuffle=False, max_len=Config.MAX_SEQ_LEN
    )
    
    # 4. Instantiate BiLSTM Model
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dense_dim=Config.DENSE_DIM,
        dropout=Config.DROPOUT,
        pad_idx=Config.PAD_IDX
    ).to(device)
    
    param_count = model.count_parameters()
    print(f"Model initialized: BiLSTMClassifier with {param_count:,} trainable parameters.")
    
    # 5. Loss, Optimizer, Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.LR_FACTOR,
        patience=Config.LR_PATIENCE
    )
    
    # 6. Training Loop with Early Stopping and Checkpointing
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    model_save_path = os.path.join(Config.MODELS_DIR, "best_bilstm_model.pt")
    
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "lr": [],
        "epoch_time_seconds": [],
        "best_epoch": 0,
        "total_train_time_seconds": 0
    }
    
    start_total_time = time.time()
    
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate_val(model, val_loader, criterion, device)
        
        epoch_time = time.time() - t0
        scheduler.step(val_loss)
        
        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 5))
        history["train_acc"].append(round(train_acc, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_acc"].append(round(val_acc, 5))
        history["val_f1"].append(round(val_f1, 5))
        history["lr"].append(current_lr)
        history["epoch_time_seconds"].append(round(epoch_time, 2))
        
        print(f"Epoch {epoch:02d}/{Config.NUM_EPOCHS:02d} [{epoch_time:.1f}s] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "config": {
                    "vocab_size": len(vocab),
                    "embedding_dim": Config.EMBEDDING_DIM,
                    "hidden_dim": Config.HIDDEN_DIM,
                    "num_layers": Config.NUM_LAYERS,
                    "dense_dim": Config.DENSE_DIM,
                    "dropout": Config.DROPOUT,
                    "max_seq_len": Config.MAX_SEQ_LEN
                }
            }, model_save_path)
            print(f"  --> Saved new best model to {model_save_path} (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  --> No improvement for {patience_counter} epoch(s) (Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch})")
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}!")
                break
                
    total_train_time = time.time() - start_total_time
    history["best_epoch"] = best_epoch
    history["total_train_time_seconds"] = round(total_train_time, 2)
    
    # Save training history
    history_path = os.path.join(Config.METRICS_DIR, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining completed in {total_train_time:.1f} seconds. History saved to {history_path}")
    return history

if __name__ == "__main__":
    train_bilstm()
