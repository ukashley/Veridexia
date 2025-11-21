"""
prepare_dataset.py:
Loads a single CSV dataset (phishing_email.csv), cleans text,
splits into train/val/test, tokenizes with DistilBERT, and
saves processed data for training.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer

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

# Detect text column
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

# Detect label column
label_col = None
for c in ["label", "class", "is_spam", "spam", "target"]:
    if c in df.columns:
        label_col = c
        break
if label_col is None:
    raise SystemExit(" No label column found. Please ensure your CSV has a label/class/spam column.")

# Map labels -> 0 legitimate, 1 phishing/spam
labels = df[label_col]
if labels.dtype == object:
    labels = labels.str.lower().map({
        "phishing": 1, "spam": 1, "legitimate": 0, "ham": 0, "not spam": 0, "nonspam": 0
    })
labels = pd.to_numeric(labels, errors="coerce")

# Build clean dataframe
df = pd.DataFrame({"text": df[text_col].astype(str), "label": labels}).dropna()
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
df["label"] = df["label"].astype(int)
print(f"[INFO] Clean dataset size: {len(df)} rows")

# Split dataset
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.125, stratify=train_df["label"], random_state=42)
print(f"[INFO] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Save splits
train_df.to_csv(PROC / "train.csv", index=False)
val_df.to_csv(PROC / "val.csv", index=False)
test_df.to_csv(PROC / "test.csv", index=False)
df.to_csv(PROC / "phishing_corpus.csv", index=False)

# Tokenization (DistilBERT)
print("[INFO] Tokenizing with DistilBERT...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

def df_to_tokenized(dataset_df, split_name):
    dataset = Dataset.from_pandas(dataset_df.reset_index(drop=True))
    dataset = dataset.map(tokenize, batched=True, remove_columns=[c for c in dataset.column_names if c not in {"text", "label"}])
    dataset.save_to_disk(str(TOK / split_name))
    print(f"[SAVED] Tokenized {split_name} -> {TOK/split_name}")

df_to_tokenized(train_df, "train")
df_to_tokenized(val_df, "val")
df_to_tokenized(test_df, "test")

print("\n DONE! Processed data saved in 'data/processed/' and tokenized datasets saved in 'data/processed/tokenized/'")
