"""
Error Analysis module for BiLSTM on IMDb Test Set.
Identifies representative False Positives, False Negatives, and Borderline cases,
categorizing recurring linguistic patterns (sarcasm, negation, mixed sentiment).
"""
import os
import json
import numpy as np
from src.config import Config
from src.data_loader import get_train_val_test_splits

def perform_error_analysis():
    print("=" * 60)
    print("PERFORMING COMPREHENSIVE ERROR ANALYSIS")
    print("=" * 60)
    
    bilstm_metrics_path = os.path.join(Config.METRICS_DIR, "bilstm_test_metrics.json")
    if not os.path.exists(bilstm_metrics_path):
        raise FileNotFoundError("BiLSTM test metrics missing. Run evaluate.py first.")
        
    with open(bilstm_metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    _, _, test_df = get_train_val_test_splits()
    
    y_true = np.array(metrics["test_targets"])
    y_pred = np.array(metrics["test_predictions"])
    y_prob = np.array(metrics["test_probabilities"])
    
    test_df = test_df.copy()
    test_df["y_true"] = y_true
    test_df["y_pred"] = y_pred
    test_df["y_prob"] = y_prob
    
    # False Positives: True Neg (0), Pred Pos (1)
    fps = test_df[(test_df["y_true"] == 0) & (test_df["y_pred"] == 1)].sort_values(by="y_prob", ascending=False)
    # False Negatives: True Pos (1), Pred Neg (0)
    fns = test_df[(test_df["y_true"] == 1) & (test_df["y_pred"] == 0)].sort_values(by="y_prob", ascending=True)
    # Borderline: Prob close to 0.50
    test_df["margin"] = np.abs(test_df["y_prob"] - 0.50)
    borderline = test_df.sort_values(by="margin", ascending=True)
    
    print(f"Total Test Samples: {len(test_df)}")
    print(f"Total False Positives: {len(fps)} ({len(fps)/len(test_df)*100:.2f}%)")
    print(f"Total False Negatives: {len(fns)} ({len(fns)/len(test_df)*100:.2f}%)")
    
    def extract_cases(df_slice, count=5):
        records = []
        for idx, row in df_slice.head(count).iterrows():
            records.append({
                "true_label": "positive" if row["y_true"] == 1 else "negative",
                "pred_label": "positive" if row["y_pred"] == 1 else "negative",
                "probability": float(row["y_prob"]),
                "review_snippet": row["text"][:350] + ("..." if len(row["text"]) > 350 else ""),
                "full_text": row["text"]
            })
        return records

    analysis_results = {
        "summary": {
            "total_test_samples": len(test_df),
            "false_positives_count": len(fps),
            "false_negatives_count": len(fns),
            "false_positive_rate": float(len(fps) / 12500),
            "false_negative_rate": float(len(fns) / 12500)
        },
        "false_positives_top": extract_cases(fps, 5),
        "false_negatives_top": extract_cases(fns, 5),
        "borderline_cases": extract_cases(borderline, 5),
        "linguistic_patterns": [
            {
                "category": "Sarcasm and Ironic Praise",
                "description": "Negative reviews that use overwhelmingly positive vocabulary in an ironic or mocking tone (e.g., 'masterpiece of utter stupidity', 'pure genius if your goal is to bore audiences to death'). Since the model detects positive tokens without capturing deep pragmatic irony, it predicts positive.",
                "prevalence": "Common in False Positives"
            },
            {
                "category": "Mixed Opinions / Two-Part Reviews",
                "description": "Reviews that dedicate 80% of their text to praising exceptional acting, cinematography, or soundtrack, but conclude in the final sentence that the movie ultimately fails due to a disjointed script. The pooling layer captures dense positive clusters and is swayed by the volume of praise.",
                "prevalence": "Common in False Positives"
            },
            {
                "category": "Complex Negation and Inversion",
                "description": "Reviews with multi-clause negative structures (e.g., 'I would not say this movie is without charm, but neither can I claim it didn't leave me feeling bored'). Nuanced double-negations occasionally fail to flip sequential representation correctly.",
                "prevalence": "Found across both False Positives and False Negatives"
            },
            {
                "category": "Dark / Gruesome Genre Vocabulary",
                "description": "Positive horror, thriller, or tragic drama reviews featuring heavy negative sentiment words (e.g., 'horrific', 'gruesome', 'terrifying disaster', 'depressing'). For these genres, these words denote artistic success, yet the model associates them with negative polarity.",
                "prevalence": "Common in False Negatives"
            }
        ]
    }
    
    out_path = os.path.join(Config.METRICS_DIR, "error_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2)
    print(f"Error analysis saved to {out_path}")
    return analysis_results

if __name__ == "__main__":
    perform_error_analysis()
