"""
Data loader module for IMDb Movie Review Dataset.
Handles loading, stratified splitting, and disk caching for fast repeated runs.
"""
import os
import glob
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import Config

def read_text_file(filepath):
    """Safely read text file with UTF-8 and latin-1 fallback."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin-1") as f:
            return f.read()

def load_raw_dataset(dataset_dir=Config.DATASET_DIR):
    """
    Load raw train and test reviews from directory structure.
    Directory structure expected:
      dataset/
        train/pos, train/neg, train/unsup
        test/pos, test/neg
    """
    train_records = []
    test_records = []
    
    for cls, label in [("pos", 1), ("neg", 0)]:
        train_path = os.path.join(dataset_dir, "train", cls)
        for filepath in glob.glob(os.path.join(train_path, "*.txt")):
            text = read_text_file(filepath)
            train_records.append({
                "text": text,
                "label": label,
                "filename": os.path.basename(filepath),
                "sentiment": cls,
                "split": "train_pool"
            })
            
        test_path = os.path.join(dataset_dir, "test", cls)
        for filepath in glob.glob(os.path.join(test_path, "*.txt")):
            text = read_text_file(filepath)
            test_records.append({
                "text": text,
                "label": label,
                "filename": os.path.basename(filepath),
                "sentiment": cls,
                "split": "test"
            })
            
    train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)
    return train_df, test_df

def get_train_val_test_splits(dataset_dir=Config.DATASET_DIR, val_size=Config.VAL_SIZE, seed=Config.RANDOM_SEED, force_reload=False):
    """
    Provides stratified train, validation, and untouched test splits.
    Uses disk caching to avoid slow file globbing on subsequent runs.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "splits.pkl")
    
    if not force_reload and os.path.exists(cache_path):
        print(f"Loading cached dataset splits from {cache_path}...")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
            return data["train"], data["val"], data["test"]
            
    print("Reading IMDb dataset from disk...")
    raw_train_df, test_df = load_raw_dataset(dataset_dir)
    
    # Stratified validation split from training pool ONLY
    train_df, val_df = train_test_split(
        raw_train_df,
        test_size=val_size,
        stratify=raw_train_df["label"],
        random_state=seed
    )
    
    train_df = train_df.copy().reset_index(drop=True)
    val_df = val_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)
    
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"train": train_df, "val": val_df, "test": test_df}, f)
        
    print(f"Dataset loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df

if __name__ == "__main__":
    tr, va, te = get_train_val_test_splits()
    print("Train class counts:\n", tr["label"].value_counts())
    print("Val class counts:\n", va["label"].value_counts())
    print("Test class counts:\n", te["label"].value_counts())
