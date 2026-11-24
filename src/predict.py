"""
Inference module for IMDb Review Sentiment Classification.
Provides a clean, reusable predictor class and CLI interface for new texts.
"""
import os
import torch
from src.config import Config
from src.preprocessing import Vocab, tokenize, clean_text
from src.model import BiLSTMClassifier

class SentimentPredictor:
    """Production-ready inference wrapper for trained BiLSTM model."""
    def __init__(self, model_path=None, vocab_path=None, device=None):
        self.device = device or Config.DEVICE
        self.model_path = model_path or os.path.join(Config.MODELS_DIR, "best_bilstm_model.pt")
        self.vocab_path = vocab_path or os.path.join(Config.MODELS_DIR, "vocab.json")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found at {self.vocab_path}")
            
        # Load vocabulary
        self.vocab = Vocab.load(self.vocab_path)
        
        # Load model checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        cfg = checkpoint.get("config", {})
        
        self.model = BiLSTMClassifier(
            vocab_size=cfg.get("vocab_size", len(self.vocab)),
            embedding_dim=cfg.get("embedding_dim", Config.EMBEDDING_DIM),
            hidden_dim=cfg.get("hidden_dim", Config.HIDDEN_DIM),
            num_layers=cfg.get("num_layers", Config.NUM_LAYERS),
            dense_dim=cfg.get("dense_dim", Config.DENSE_DIM),
            dropout=cfg.get("dropout", Config.DROPOUT),
            pad_idx=Config.PAD_IDX
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, raw_text: str) -> dict:
        """
        Classify sentiment for raw review text.
        Applies exact training preprocessing, tokenization, and padding.
        """
        if not raw_text or not raw_text.strip():
            return {
                "sentiment": "neutral/invalid",
                "probability": 0.50,
                "confidence": 0.0,
                "label": -1,
                "tokens_count": 0,
                "error": "Empty or whitespace-only input."
            }
            
        cleaned = clean_text(raw_text)
        tokens = tokenize(raw_text)
        token_ids = self.vocab.encode(tokens, max_len=Config.MAX_SEQ_LEN)
        
        input_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            logit = self.model(input_tensor)
            prob_pos = float(torch.sigmoid(logit).item())
            
        sentiment = "positive" if prob_pos >= 0.50 else "negative"
        confidence = prob_pos if sentiment == "positive" else (1.0 - prob_pos)
        
        return {
            "sentiment": sentiment,
            "probability_positive": round(prob_pos, 4),
            "probability_negative": round(1.0 - prob_pos, 4),
            "confidence": round(confidence * 100, 2),
            "label": 1 if sentiment == "positive" else 0,
            "tokens_count": len(tokens),
            "cleaned_sample": cleaned[:120] + ("..." if len(cleaned) > 120 else "")
        }

if __name__ == "__main__":
    predictor = SentimentPredictor()
    test_samples = [
        "This movie was an absolute masterpiece! Incredible acting, brilliant directing, and a deeply emotional story.",
        "What a complete waste of time. The plot made no sense, the acting was dreadful, and I almost fell asleep.",
        "Not bad, but not particularly great either. It had some entertaining moments despite pacing issues.",
        "I didn't expect much, but I was pleasantly surprised by how gripping and thoughtful this film turned out to be."
    ]
    
    print("\n--- SAMPLE PREDICTIONS ---")
    for sample in test_samples:
        res = predictor.predict(sample)
        print(f"\nReview: '{sample}'")
        print(f"Sentiment: {res['sentiment'].upper()} (Confidence: {res['confidence']:.1f}%, P(pos): {res['probability_positive']})")
