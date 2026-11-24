"""
Baseline Model: TF-IDF + Logistic Regression.
Provides a standard, competitive benchmark to contextualize BiLSTM performance.
"""
import os
import json
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from src.config import Config
from src.data_loader import get_train_val_test_splits
from src.preprocessing import clean_text

def run_baseline():
    print("=" * 60)
    print("RUNNING BASELINE: TF-IDF + Logistic Regression")
    print("=" * 60)
    
    train_df, val_df, test_df = get_train_val_test_splits()
    
    print("Cleaning text for TF-IDF...")
    train_clean = [clean_text(t) for t in train_df["text"]]
    val_clean = [clean_text(t) for t in val_df["text"]]
    test_clean = [clean_text(t) for t in test_df["text"]]
    
    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values
    
    # Vectorizer fitted ONLY on training split
    print(f"Fitting TfidfVectorizer (max_features={Config.VOCAB_SIZE}, ngram_range=(1,2))...")
    vectorizer = TfidfVectorizer(
        max_features=Config.VOCAB_SIZE,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(train_clean)
    X_val = vectorizer.transform(val_clean)
    X_test = vectorizer.transform(test_clean)
    
    print("Training LogisticRegression (C=1.0, max_iter=1000)...")
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=Config.RANDOM_SEED)
    clf.fit(X_train, y_train)
    
    # Predict on validation
    val_preds = clf.predict(X_val)
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_acc = accuracy_score(y_val, val_preds)
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    # Predict on untouched test set
    test_preds = clf.predict(X_test)
    test_probs = clf.predict_proba(X_test)[:, 1]
    
    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds)
    test_rec = recall_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs)
    cm = confusion_matrix(y_test, test_preds).tolist()
    report = classification_report(y_test, test_preds, target_names=["neg", "pos"], output_dict=True)
    
    print("\n--- BASELINE TEST METRICS ---")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall:    {test_rec:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    print(f"ROC-AUC:   {test_auc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds))
    
    # Save model and vectorizer
    model_save_path = os.path.join(Config.MODELS_DIR, "baseline_tfidf_lr.joblib")
    joblib.dump({"vectorizer": vectorizer, "classifier": clf}, model_save_path)
    print(f"Baseline model saved to {model_save_path}")
    
    # Save metrics
    metrics = {
        "model_name": "TF-IDF + Logistic Regression",
        "accuracy": float(test_acc),
        "precision": float(test_prec),
        "recall": float(test_rec),
        "f1_score": float(test_f1),
        "roc_auc": float(test_auc),
        "confusion_matrix": cm,
        "classification_report": report,
        "test_probabilities": test_probs.tolist(),
        "test_predictions": test_preds.tolist()
    }
    metrics_path = os.path.join(Config.METRICS_DIR, "baseline_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Baseline metrics saved to {metrics_path}")
    
    return metrics

if __name__ == "__main__":
    run_baseline()
