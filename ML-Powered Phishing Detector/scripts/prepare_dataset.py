"""
prepare_dataset.py:
Loads a single CSV dataset (phishing_email.csv), cleans text,
splits into train/val/test, tokenizes with DistilBERT and
saves processed data for training.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re
from html import unescape
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer
import json

# Paths
RAW = Path("data/phishing_email.csv")
PROC = Path("data/processed")
TOK = PROC / "tokenized"
PROC.mkdir(parents=True, exist_ok=True)
TOK.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Loading dataset from {RAW}")
df = pd.read_csv(RAW, encoding="utf-8", on_bad_lines="skip")
df.columns = [c.lower().strip() for c in df.columns]
print(f"[INFO] Columns detected: {list(df.columns)}")
print(f"[INFO] Initial rows: {len(df)}")

# Try a few common column names first so the script can cope with small dataset
# variations without needing manual edits every time.
text_col = None
for c in ["text", "email_text", "message", "content", "body"]:
    if c in df.columns:
        text_col = c
        break
if text_col is None:
    # combine subject + body if separate
    if {"subject", "body"}.issubset(df.columns):
        df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).astype(str)
        text_col = "text"
    else:
        # fallback: concatenate all string columns
        df["text"] = df.select_dtypes(include="object").astype(str).agg(" ".join, axis=1)
        text_col = "text"

# To keep the script forgiving enough for slightly different CSV schemas, 
# but still fail loudly if nothing usable exists.
label_col = None
for c in ["label", "class", "is_spam", "spam", "target"]:
    if c in df.columns:
        label_col = c
        break
if label_col is None:
    raise SystemExit(" No label column found. Please ensure your CSV has a label/class/spam column.")

# Normalise label values into the exact 0/1 scheme expected by the training code.
labels = df[label_col]
if labels.dtype == object:
   labels = labels.str.lower().str.strip().map({
        "phishing": 1, "spam": 1, "legitimate": 0, "ham": 0, 
        "not spam": 0, "nonspam": 0, "safe": 0, "unsafe": 1
    })
labels = pd.to_numeric(labels, errors="coerce")

# To keep only the fields the downstream pipeline actually needs.
df = pd.DataFrame({"text": df[text_col].astype(str), "label": labels}).dropna()
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
df["label"] = df["label"].astype(int)

print(f"[INFO] After initial cleaning: {len(df)} rows")

# ===== HTML CLEANING & TEXT PREPROCESSING =====
print("\n[INFO] Cleaning HTML and preprocessing text...")

def clean_text(text):
    """Remove HTML, decode entities, normalize whitespace"""
    text = str(text)
    # Decode HTML entities
    text = unescape(text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove URLs (but keep them for feature extraction first)
    # text = re.sub(r'http[s]?://\S+', ' URL ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text'] = df['text'].apply(clean_text)

# ===== DATA QUALITY VALIDATION =====
print("\n[INFO] Performing data quality checks...")

df['text_length'] = df['text'].str.len()
print(f"[INFO] Text length stats:")
print(f"  Min: {df['text_length'].min()}")
print(f"  Max: {df['text_length'].max()}")
print(f"  Mean: {df['text_length'].mean():.1f}")
print(f"  Median: {df['text_length'].median():.1f}")

# Filter suspicious lengths
before_filter = len(df)
df = df[(df['text_length'] >= 20) & (df['text_length'] <= 10000)].copy()
after_filter = len(df)
if before_filter > after_filter:
    print(f"[INFO] Filtered {before_filter - after_filter} emails (too short/long)")

df = df.drop('text_length', axis=1)

# ===== CLASS DISTRIBUTION =====
print("\n[INFO] Checking class distribution...")
label_counts = df['label'].value_counts()
label_distribution = df['label'].value_counts(normalize=True)

print(f"[INFO] Label counts:")
print(f"  Legitimate (0): {label_counts.get(0, 0)}")
print(f"  Phishing (1): {label_counts.get(1, 0)}")
print(f"[INFO] Label distribution:")
print(f"  Legitimate: {label_distribution.get(0, 0):.2%}")
print(f"  Phishing: {label_distribution.get(1, 0):.2%}")

minority_share = min(label_distribution.get(0, 0), label_distribution.get(1, 0))
majority_count = max(label_counts.get(0, 0), label_counts.get(1, 0))
minority_count = min(label_counts.get(0, 0), label_counts.get(1, 0))
imbalance_ratio = (majority_count / minority_count) if minority_count else None

if minority_share < 0.2:
    print(f"\n  WARNING: Severe class imbalance! Minority class: {minority_share:.2%}")
    print("   -> Consider class weights in training")

# ===== FEATURE ENGINEERING =====
print("\n[INFO] Extracting features for analysis...")

def count_urls(text):
    """Count URLs in text"""
    return len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))

def urgency_score(text):
    """Score based on urgency keywords"""
    keywords = [
        'urgent', 'immediate', 'action required', 'verify now', 'suspended',
        'expire', 'limited time', 'act now', 'confirm', 'security alert',
        'unusual activity', 'locked', 'verify your account', 'click here',
        'update required', 'suspended account', 'verify identity'
    ]
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)

def suspicious_patterns(text):
    """Detect common phishing patterns"""
    patterns = [
        r'dear (customer|user|member|sir|madam)',
        r'verify (your|account|identity)',
        r'password (reset|update|expire|change)',
        r'click (here|below|link)',
        r'account (suspended|locked|compromised)',
        r'confirm (identity|account|information)',
        r'security (alert|warning|notice)',
        r'unusual (activity|sign-in|login)'
    ]
    text_lower = text.lower()
    return sum(1 for pattern in patterns if re.search(pattern, text_lower))

def count_special_chars(text):
    """Count special characters (phishing often has odd formatting)"""
    return len(re.findall(r'[!@#$%^&*()]', text))

# Features saved for analysis/reporting even though the live app now
# relies more on model output plus supporting rule evidence than on a single handcrafted score.
df['url_count'] = df['text'].apply(count_urls)
df['urgency_score'] = df['text'].apply(urgency_score)
df['suspicious_patterns'] = df['text'].apply(suspicious_patterns)
df['special_char_count'] = df['text'].apply(count_special_chars)

print(f"\n[INFO] Feature statistics:")
print(f"  URLs - Phishing: {df[df['label']==1]['url_count'].mean():.2f}, Legitimate: {df[df['label']==0]['url_count'].mean():.2f}")
print(f"  Urgency - Phishing: {df[df['label']==1]['urgency_score'].mean():.2f}, Legitimate: {df[df['label']==0]['urgency_score'].mean():.2f}")
print(f"  Suspicious patterns - Phishing: {df[df['label']==1]['suspicious_patterns'].mean():.2f}, Legitimate: {df[df['label']==0]['suspicious_patterns'].mean():.2f}")

print(f"\n[INFO] Final clean dataset: {len(df)} rows")

# Split once with stratification so the saved CSVs and tokenized datasets stay aligned.
print("\n[INFO] Splitting dataset (70% train, 10% val, 20% test)...")
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.125, stratify=train_df["label"], random_state=42)

print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

# Save splits
train_df.to_csv(PROC / "train.csv", index=False)
val_df.to_csv(PROC / "val.csv", index=False)
test_df.to_csv(PROC / "test.csv", index=False)
df.to_csv(PROC / "phishing_corpus.csv", index=False)
print(f"\n[SAVED] CSV splits -> {PROC}/")

# The tokenized datasets are saved to disk so training does not have to redo
# preprocessing every time the notebook or script is rerun.
print("\n[INFO] Tokenizing with DistilBERT...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

def df_to_tokenized(dataset_df, split_name):
   # Keeping text and label only for tokenization
    dataset = Dataset.from_pandas(dataset_df[['text', 'label']].reset_index(drop=True))
    dataset = dataset.map(tokenize, batched=True, remove_columns=['text'])
    dataset.save_to_disk(str(TOK / split_name))
    print(f"[SAVED] {split_name} -> {TOK/split_name}")

df_to_tokenized(train_df, "train")
df_to_tokenized(val_df, "val")
df_to_tokenized(test_df, "test")

# Save a small JSON summary because the app and report both reuse these values later.
print("\n[INFO] Saving dataset statistics...")
stats = {
    "dataset_info": {
        "total_samples": len(df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df)
    },
    "class_distribution": {
        "legitimate_count": int(label_counts.get(0, 0)),
        "phishing_count": int(label_counts.get(1, 0)),
        "legitimate_ratio": float(label_distribution.get(0, 0)),
        "phishing_ratio": float(label_distribution.get(1, 0)),
        "imbalance_ratio": float(imbalance_ratio) if imbalance_ratio is not None else None
    },
    "feature_statistics": {
        "avg_urls_phishing": float(df[df['label']==1]['url_count'].mean()),
        "avg_urls_legitimate": float(df[df['label']==0]['url_count'].mean()),
        "avg_urgency_phishing": float(df[df['label']==1]['urgency_score'].mean()),
        "avg_urgency_legitimate": float(df[df['label']==0]['urgency_score'].mean()),
        "avg_suspicious_phishing": float(df[df['label']==1]['suspicious_patterns'].mean()),
        "avg_suspicious_legitimate": float(df[df['label']==0]['suspicious_patterns'].mean())
    }
}

with open(PROC / "dataset_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"[SAVED] Statistics -> {PROC}/dataset_stats.json")

print("\n" + "="*60)
print(" DATASET PREPARATION COMPLETE!")
print("="*60)
print(f" Summary:")
print(f"  - Total samples: {len(df):,}")
print(f"  - Legitimate: {label_counts.get(0, 0):,} ({label_distribution.get(0, 0):.1%})")
print(f"  - Phishing: {label_counts.get(1, 0):,} ({label_distribution.get(1, 0):.1%})")
print(f"  - Features extracted: url_count, urgency_score, suspicious_patterns")
print(f"  - Tokenized for DistilBERT training")
print(f"\n Output locations:")
print(f"  - CSV splits: {PROC}/")
print(f"  - Tokenized data: {TOK}/")
print(f"  - Statistics: {PROC}/dataset_stats.json")
print("\n Ready for training!")
