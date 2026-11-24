"""
Text preprocessing and Tokenization module.
Handles text cleaning, tokenization, vocabulary building, padding/truncation,
and PyTorch Dataset / DataLoader generation.
"""
import re
import html
import json
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from src.config import Config

# Standard negation mapping to preserve critical sentiment cues
CONTRACTIONS = {
    r"\bcan\'t\b": "can not",
    r"\bcannot\b": "can not",
    r"\bwon\'t\b": "will not",
    r"\bn\'t\b": " not",
    r"\b\'re\b": " are",
    r"\b\'s\b": " is",
    r"\b\'d\b": " would",
    r"\b\'ll\b": " will",
    r"\b\'ve\b": " have",
    r"\b\'m\b": " am"
}

def clean_text(text: str) -> str:
    """
    Cleans raw review text:
    - Decodes HTML entities and handles <br /> tags
    - Expands negations and contractions
    - Preserves sentiment-bearing words and basic punctuation
    - Normalizes excessive whitespace
    - Converts to lowercase
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Unescape HTML entities (e.g., &amp; -> &, &quot; -> ")
    text = html.unescape(text)
    
    # 2. Replace HTML line breaks with a space
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    
    # 3. Remove remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    
    # 4. Normalize contractions (especially negations like isn't -> is not)
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
    # 5. Lowercase
    text = text.lower()
    
    # 6. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> list[str]:
    """
    Tokenizes cleaned text into words and punctuation tokens.
    Regex matches words (including hyphenated or apostrophed) and standalone punctuation.
    """
    cleaned = clean_text(text)
    tokens = re.findall(r"\b[\w'-]+\b|[!?,.:;]", cleaned)
    return tokens

class Vocab:
    """
    Vocabulary container with token-to-index and index-to-token mappings.
    Fitted exclusively on the training subset to prevent data leakage.
    """
    def __init__(self, pad_token=Config.PAD_TOKEN, unk_token=Config.UNK_TOKEN,
                 pad_idx=Config.PAD_IDX, unk_idx=Config.UNK_IDX):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        
        self.token2idx = {pad_token: pad_idx, unk_token: unk_idx}
        self.idx2token = {pad_idx: pad_token, unk_idx: unk_token}
        self.frequencies = Counter()
        self.fitted = False

    def build_vocab(self, texts: list[str], max_size=Config.VOCAB_SIZE, min_freq=Config.MIN_FREQ):
        """Build vocabulary from a list of raw review texts."""
        print(f"Building vocabulary from {len(texts)} training texts...")
        self.frequencies = Counter()
        for text in texts:
            tokens = tokenize(text)
            self.frequencies.update(tokens)
            
        # Filter by min_freq and limit to max_size
        most_common = self.frequencies.most_common()
        idx = len(self.token2idx)
        for token, count in most_common:
            if count < min_freq:
                break
            if len(self.token2idx) >= max_size:
                break
            if token not in self.token2idx:
                self.token2idx[token] = idx
                self.idx2token[idx] = token
                idx += 1
                
        self.fitted = True
        print(f"Vocabulary built: {len(self.token2idx)} unique tokens (capped at {max_size}, min_freq={min_freq}).")

    def encode(self, tokens: list[str], max_len=Config.MAX_SEQ_LEN) -> list[int]:
        """
        Maps tokens to integer IDs with post-truncation and post-padding.
        """
        token_ids = [self.token2idx.get(t, self.unk_idx) for t in tokens]
        
        # Post-truncation
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
            
        # Post-padding
        if len(token_ids) < max_len:
            token_ids = token_ids + [self.pad_idx] * (max_len - len(token_ids))
            
        return token_ids

    def save(self, filepath: str):
        """Save vocabulary mappings and frequency stats to JSON."""
        data = {
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "pad_idx": self.pad_idx,
            "unk_idx": self.unk_idx,
            "token2idx": self.token2idx,
            "top_tokens": self.frequencies.most_common(100),
            "vocab_size": len(self.token2idx)
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Vocabulary saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load vocabulary from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(
            pad_token=data["pad_token"],
            unk_token=data["unk_token"],
            pad_idx=data["pad_idx"],
            unk_idx=data["unk_idx"]
        )
        vocab.token2idx = data["token2idx"]
        vocab.idx2token = {int(v): k for k, v in vocab.token2idx.items()}
        vocab.fitted = True
        return vocab

    def __len__(self):
        return len(self.token2idx)

class IMDbDataset(Dataset):
    """PyTorch Dataset for IMDb reviews."""
    def __init__(self, texts: list[str], labels: list[int], vocab: Vocab, max_len=Config.MAX_SEQ_LEN):
        self.labels = list(labels)
        self.encoded_sequences = []
        
        for text in texts:
            tokens = tokenize(text)
            encoded = vocab.encode(tokens, max_len=max_len)
            self.encoded_sequences.append(encoded)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.encoded_sequences[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

def get_dataloader(texts, labels, vocab, batch_size=Config.BATCH_SIZE, shuffle=False, max_len=Config.MAX_SEQ_LEN):
    """Creates a PyTorch DataLoader."""
    dataset = IMDbDataset(texts, labels, vocab, max_len=max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
