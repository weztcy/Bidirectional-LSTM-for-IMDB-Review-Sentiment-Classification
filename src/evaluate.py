"""
Evaluation script for the BiLSTM model on the untouched test set.
Computes accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix,
and per-class classification report.
"""
import os
import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from src.config import Config
from src.data_loader import get_train_val_test_splits
from src.preprocessing import Vocab, get_dataloader
from src.model import BiLSTMClassifier

def evaluate_bilstm_test():
    print("=" * 60)
    print("EVALUATING BiLSTM ON UNTOUCHED TEST SET")
    print("=" * 60)
    
    device = Config.DEVICE
    model_path = os.path.join(Config.MODELS_DIR, "best_bilstm_model.pt")
    vocab_path = os.path.join(Config.MODELS_DIR, "vocab.json")
    
    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        raise FileNotFoundError("Model or Vocab file missing. Run training first.")
        
    vocab = Vocab.load(vocab_path)
    checkpoint = torch.load(model_path, map_location=device)
    cfg = checkpoint.get("config", {})
    
    model = BiLSTMClassifier(
        vocab_size=cfg.get("vocab_size", len(vocab)),
        embedding_dim=cfg.get("embedding_dim", Config.EMBEDDING_DIM),
        hidden_dim=cfg.get("hidden_dim", Config.HIDDEN_DIM),
        num_layers=cfg.get("num_layers", Config.NUM_LAYERS),
        dense_dim=cfg.get("dense_dim", Config.DENSE_DIM),
        dropout=cfg.get("dropout", Config.DROPOUT),
        pad_idx=Config.PAD_IDX
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded best BiLSTM model checkpoint from epoch {checkpoint.get('epoch')} (Val Loss: {checkpoint.get('val_loss'):.4f})")
    
    _, _, test_df = get_train_val_test_splits()
    print(f"Test set loaded: {len(test_df)} samples (Pos: {(test_df['label'] == 1).sum()}, Neg: {(test_df['label'] == 0).sum()})")
    
    test_loader = get_dataloader(
        test_df["text"].tolist(), test_df["label"].tolist(),
        vocab, batch_size=Config.BATCH_SIZE, shuffle=False, max_len=Config.MAX_SEQ_LEN
    )
    
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_probs = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.50).astype(int)
            
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_targets.extend(batch_y.cpu().numpy().tolist())
            
    test_loss = total_loss / len(test_loader.dataset)
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=["neg", "pos"], output_dict=True)
    
    print("\n" + "=" * 60)
    print("FINAL BiLSTM TEST SET PERFORMANCE")
    print("=" * 60)
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall:    {rec:.4f}")
    print(f"Test F1-Score:  {f1:.4f}")
    print(f"Test ROC-AUC:   {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"  TN: {cm[0][0]:<6} FP: {cm[0][1]:<6}")
    print(f"  FN: {cm[1][0]:<6} TP: {cm[1][1]:<6}")
    
    results = {
        "model_name": "Bidirectional LSTM (BiLSTM)",
        "test_loss": float(test_loss),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm,
        "classification_report": report,
        "test_probabilities": all_probs,
        "test_predictions": all_preds,
        "test_targets": all_targets
    }
    
    out_path = os.path.join(Config.METRICS_DIR, "bilstm_test_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nBiLSTM test metrics saved to {out_path}")
    return results

if __name__ == "__main__":
    evaluate_bilstm_test()
