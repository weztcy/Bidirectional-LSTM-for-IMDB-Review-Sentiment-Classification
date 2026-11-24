"""
Visualization module generating all mandatory figures for the final report.
Outputs high-resolution PNG plots to outputs/figures/.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

from src.config import Config
from src.data_loader import get_train_val_test_splits
from src.preprocessing import clean_text

# Configure global publication-quality plot aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

PRIMARY_COLOR = "#2563EB"   # Deep Blue (BiLSTM)
SECONDARY_COLOR = "#D97706" # Warm Amber (Baseline)
ACCENT_GREEN = "#059669"
ACCENT_RED = "#DC2626"

def plot_class_distribution(train_df, val_df, test_df):
    """Figure 1: Class distribution across train, validation, and test sets."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    data = {
        "Train": [sum(train_df["label"] == 0), sum(train_df["label"] == 1)],
        "Validation": [sum(val_df["label"] == 0), sum(val_df["label"] == 1)],
        "Test": [sum(test_df["label"] == 0), sum(test_df["label"] == 1)]
    }
    
    x = np.arange(len(data))
    width = 0.35
    
    neg_counts = [data[k][0] for k in data]
    pos_counts = [data[k][1] for k in data]
    
    bars1 = ax.bar(x - width/2, neg_counts, width, label="Negative (0)", color="#EF4444", alpha=0.85, edgecolor="#991B1B")
    bars2 = ax.bar(x + width/2, pos_counts, width, label="Positive (1)", color="#10B981", alpha=0.85, edgecolor="#065F46")
    
    ax.set_ylabel("Number of Reviews", fontweight="bold")
    ax.set_title("Class Distribution Across Dataset Splits", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(data.keys(), fontweight="bold")
    ax.legend(frameon=True, loc="upper right")
    ax.set_ylim(0, max(neg_counts) * 1.15)
    
    # Add count labels on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 150, f"{yval:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 150, f"{yval:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    save_path = os.path.join(Config.FIGURES_DIR, "class_distribution.png")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")

def plot_review_length_distribution(train_df):
    """Figure 2: Review length distribution with sequence cutoff indication."""
    fig, ax = plt.subplots(figsize=(9, 5))
    
    lengths_neg = [len(clean_text(t).split()) for t in train_df[train_df["label"] == 0]["text"]]
    lengths_pos = [len(clean_text(t).split()) for t in train_df[train_df["label"] == 1]["text"]]
    all_lengths = lengths_neg + lengths_pos
    
    p50 = np.percentile(all_lengths, 50)
    p75 = np.percentile(all_lengths, 75)
    p90 = np.percentile(all_lengths, 90)
    p95 = np.percentile(all_lengths, 95)
    cutoff = Config.MAX_SEQ_LEN
    coverage = np.mean(np.array(all_lengths) <= cutoff) * 100
    
    bins = np.linspace(0, 1000, 60)
    ax.hist(lengths_neg, bins=bins, alpha=0.5, color="#EF4444", label="Negative Reviews", density=True)
    ax.hist(lengths_pos, bins=bins, alpha=0.5, color="#10B981", label="Positive Reviews", density=True)
    
    ax.axvline(p50, color="#4B5563", linestyle="--", linewidth=1.5, label=f"Median ({int(p50)} words)")
    ax.axvline(cutoff, color="#1D4ED8", linestyle="-", linewidth=2.2, label=f"Max Seq Len = {cutoff} ({coverage:.1f}% coverage)")
    
    ax.set_title("Training Set Review Length Distribution (Word Count)", fontweight="bold", pad=12)
    ax.set_xlabel("Review Length (Words)", fontweight="bold")
    ax.set_ylabel("Probability Density", fontweight="bold")
    ax.set_xlim(0, 1000)
    ax.legend(frameon=True, loc="upper right")
    
    save_path = os.path.join(Config.FIGURES_DIR, "review_length_distribution.png")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")

def plot_training_history(history):
    """Figure 3 & 4: Training vs Validation Loss and Accuracy Curves."""
    epochs = history["epoch"]
    best_epoch = history.get("best_epoch", 1)
    
    # 1. Loss history
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], "o-", color="#2563EB", linewidth=2.2, label="Training Loss")
    ax.plot(epochs, history["val_loss"], "s-", color="#DC2626", linewidth=2.2, label="Validation Loss")
    ax.axvline(best_epoch, color="#059669", linestyle="--", linewidth=1.8, label=f"Best Model (Epoch {best_epoch})")
    
    ax.set_title("BiLSTM Training and Validation Loss Across Epochs", fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Binary Cross-Entropy Loss", fontweight="bold")
    ax.set_xticks(epochs)
    ax.legend(frameon=True, loc="upper right")
    
    save_path = os.path.join(Config.FIGURES_DIR, "loss_history.png")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")
    
    # 2. Accuracy history
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_acc"], "o-", color="#2563EB", linewidth=2.2, label="Training Accuracy")
    ax.plot(epochs, history["val_acc"], "s-", color="#DC2626", linewidth=2.2, label="Validation Accuracy")
    ax.axvline(best_epoch, color="#059669", linestyle="--", linewidth=1.8, label=f"Best Model (Epoch {best_epoch})")
    
    ax.set_title("BiLSTM Training and Validation Accuracy Across Epochs", fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Accuracy", fontweight="bold")
    ax.set_xticks(epochs)
    ax.set_ylim(0.70, 1.0)
    ax.legend(frameon=True, loc="lower right")
    
    save_path = os.path.join(Config.FIGURES_DIR, "accuracy_history.png")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")

def plot_confusion_matrix(cm, model_name="BiLSTM"):
    """Figure 5: Confusion matrix with counts and percentages."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cm_arr = np.array(cm)
    cm_pct = cm_arr / cm_arr.sum() * 100
    
    annot = np.array([
        [f"{cm_arr[0,0]:,}\n({cm_pct[0,0]:.1f}%)", f"{cm_arr[0,1]:,}\n({cm_pct[0,1]:.1f}%)"],
        [f"{cm_arr[1,0]:,}\n({cm_pct[1,0]:.1f}%)", f"{cm_arr[1,1]:,}\n({cm_pct[1,1]:.1f}%)"]
    ])
    
    sns.heatmap(cm_arr, annot=annot, fmt="", cmap="Blues", cbar=True, ax=ax,
                xticklabels=["Negative (0)", "Positive (1)"],
                yticklabels=["Negative (0)", "Positive (1)"],
                linewidths=1.2, linecolor="white", annot_kws={"size": 13, "fontweight": "bold"})
    
    ax.set_title(f"Test Set Confusion Matrix ({model_name})", fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontweight="bold")
    ax.set_ylabel("True Class", fontweight="bold")
    
    save_path = os.path.join(Config.FIGURES_DIR, "confusion_matrix.png")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")

def plot_roc_pr_curves(bilstm_metrics, baseline_metrics):
    """Figure 6 & 7: ROC and Precision-Recall Curves comparing BiLSTM vs Baseline."""
    y_true = np.array(bilstm_metrics["test_targets"])
    bilstm_probs = np.array(bilstm_metrics["test_probabilities"])
    base_probs = np.array(baseline_metrics["test_probabilities"])
    
    # 1. ROC Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    
    fpr_bi, tpr_bi, _ = roc_curve(y_true, bilstm_probs)
    auc_bi = auc(fpr_bi, tpr_bi)
    
    fpr_base, tpr_base, _ = roc_curve(y_true, base_probs)
    auc_base = auc(fpr_base, tpr_base)
    
    ax.plot(fpr_bi, tpr_bi, color=PRIMARY_COLOR, linewidth=2.4, label=f"BiLSTM (AUC = {auc_bi:.4f})")
    ax.plot(fpr_base, tpr_base, color=SECONDARY_COLOR, linewidth=2.0, linestyle="--", label=f"TF-IDF + LR (AUC = {auc_base:.4f})")
    ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle=":", linewidth=1.5, label="Random Chance (AUC = 0.5000)")
    
    ax.set_title("Test Set ROC Curves Comparison", fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall)", fontweight="bold")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(frameon=True, loc="lower right")
    
    save_path = os.path.join(Config.FIGURES_DIR, "roc_curve.png")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")
    
    # 2. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    
    prec_bi, rec_bi, _ = precision_recall_curve(y_true, bilstm_probs)
    ap_bi = average_precision_score(y_true, bilstm_probs)
    
    prec_base, rec_base, _ = precision_recall_curve(y_true, base_probs)
    ap_base = average_precision_score(y_true, base_probs)
    
    ax.plot(rec_bi, prec_bi, color=PRIMARY_COLOR, linewidth=2.4, label=f"BiLSTM (AP = {ap_bi:.4f})")
    ax.plot(rec_base, prec_base, color=SECONDARY_COLOR, linewidth=2.0, linestyle="--", label=f"TF-IDF + LR (AP = {ap_base:.4f})")
    
    ax.set_title("Test Set Precision-Recall Curves Comparison", fontweight="bold", pad=12)
    ax.set_xlabel("Recall", fontweight="bold")
    ax.set_ylabel("Precision", fontweight="bold")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0.70, 1.02])
    ax.legend(frameon=True, loc="lower left")
    
    save_path = os.path.join(Config.FIGURES_DIR, "precision_recall_curve.png")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")

def plot_model_comparison(bilstm_metrics, baseline_metrics):
    """Figure 8: Comparative bar chart across multiple evaluation metrics."""
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    bilstm_scores = [
        bilstm_metrics["accuracy"],
        bilstm_metrics["precision"],
        bilstm_metrics["recall"],
        bilstm_metrics["f1_score"],
        bilstm_metrics["roc_auc"]
    ]
    baseline_scores = [
        baseline_metrics["accuracy"],
        baseline_metrics["precision"],
        baseline_metrics["recall"],
        baseline_metrics["f1_score"],
        baseline_metrics["roc_auc"]
    ]
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, bilstm_scores, width, label="BiLSTM (Deep Learning)", color=PRIMARY_COLOR, alpha=0.9)
    bars2 = ax.bar(x + width/2, baseline_scores, width, label="TF-IDF + LR (Baseline)", color=SECONDARY_COLOR, alpha=0.9)
    
    ax.set_title("Model Performance Comparison on IMDb Test Set", fontweight="bold", pad=12)
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontweight="bold")
    ax.set_ylim(0.70, 1.0)
    ax.legend(frameon=True, loc="upper right")
    
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.007, f"{yval:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.007, f"{yval:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        
    save_path = os.path.join(Config.FIGURES_DIR, "model_comparison.png")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")

def plot_vocab_top_tokens():
    """Figure 9: Top 25 most frequent tokens in vocabulary."""
    vocab_path = os.path.join(Config.MODELS_DIR, "vocab.json")
    if not os.path.exists(vocab_path):
        return
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    top_tokens = data.get("top_tokens", [])[:25]
    if not top_tokens:
        return
        
    tokens, counts = zip(*reversed(top_tokens))
    
    fig, ax = plt.subplots(figsize=(8, 7))
    y_pos = np.arange(len(tokens))
    ax.barh(y_pos, counts, color="#3B82F6", alpha=0.85, edgecolor="#1D4ED8")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tokens, fontsize=9.5)
    ax.set_xlabel("Corpus Frequency", fontweight="bold")
    ax.set_title("Top 25 Most Frequent Tokens in Training Corpus", fontweight="bold", pad=12)
    
    save_path = os.path.join(Config.FIGURES_DIR, "vocab_frequent_tokens.png")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")

def generate_all_visualizations():
    print("=" * 60)
    print("GENERATING ALL FIGURES")
    print("=" * 60)
    
    train_df, val_df, test_df = get_train_val_test_splits()
    plot_class_distribution(train_df, val_df, test_df)
    plot_review_length_distribution(train_df)
    
    history_path = os.path.join(Config.METRICS_DIR, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        plot_training_history(history)
        
    bilstm_metrics_path = os.path.join(Config.METRICS_DIR, "bilstm_test_metrics.json")
    baseline_metrics_path = os.path.join(Config.METRICS_DIR, "baseline_metrics.json")
    
    if os.path.exists(bilstm_metrics_path):
        with open(bilstm_metrics_path, "r", encoding="utf-8") as f:
            bilstm_metrics = json.load(f)
        plot_confusion_matrix(bilstm_metrics["confusion_matrix"], "BiLSTM")
        
        if os.path.exists(baseline_metrics_path):
            with open(baseline_metrics_path, "r", encoding="utf-8") as f:
                baseline_metrics = json.load(f)
            plot_roc_pr_curves(bilstm_metrics, baseline_metrics)
            plot_model_comparison(bilstm_metrics, baseline_metrics)
            
    plot_vocab_top_tokens()
    print("All visualizations generated successfully!")

if __name__ == "__main__":
    generate_all_visualizations()
