from pathlib import Path
import pandas as pd
import csv, sys

# To allow for very large email bodies
csv.field_size_limit(sys.maxsize)

IN_PATH  = Path("data/TREC-06.csv")          
OUT_PATH = Path("data/processed/trec06_processed.csv")

df = pd.read_csv(IN_PATH, engine="python")

# Drop rows with no label
df = df.dropna(subset=["label"]).copy()
df["label"] = df["label"].astype(int)

df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).astype(str)

# Keep only what your pipeline expects
out = df[["text", "label"]].copy()

# Basic clean up to prevent whitespace 
out["text"] = out["text"].str.replace(r"\s+", " ", regex=True).str.strip()

# Save
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_PATH, index=False, encoding="utf-8")

print("Saved:", OUT_PATH)
print("Rows:", len(out))
print("Label counts:\n", out["label"].value_counts())
