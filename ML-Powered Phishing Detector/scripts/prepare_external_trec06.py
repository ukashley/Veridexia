from pathlib import Path
import pandas as pd
import csv, sys

# TREC-06 can contain long message bodies, so raise the CSV field limit first.
csv.field_size_limit(sys.maxsize)

IN_PATH  = Path("data/TREC-06.csv")          
OUT_PATH = Path("data/processed/trec06_processed.csv")

df = pd.read_csv(IN_PATH, engine="python")

# Drop rows with no label because external evaluation needs known ground truth.
df = df.dropna(subset=["label"]).copy()
df["label"] = df["label"].astype(int)

# Match the internal model input shape by combining subject and body into one text field.
df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).astype(str)

# Strip the export back to the two fields the evaluators actually need.
out = df[["text", "label"]].copy()

# Basic clean up to prevent whitespace 
out["text"] = out["text"].str.replace(r"\s+", " ", regex=True).str.strip()

# Save
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_PATH, index=False, encoding="utf-8")

print("Saved:", OUT_PATH)
print("Rows:", len(out))
print("Label counts:\n", out["label"].value_counts())
