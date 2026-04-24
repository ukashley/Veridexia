from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

MODEL_DIR = Path("models/distilbert/final_model")
DATA_PATH = Path("data/processed/trec06_processed.csv")

# This script is lightweight so it can load the prepared external dataset, find the
# usable text column, then run batched inference over the saved model.
df = pd.read_csv(DATA_PATH)

print("Columns:", list(df.columns))  

# To pick the right text column
TEXT_COL_CANDIDATES = ["text_combined", "text", "email", "body"]
text_col = next((c for c in TEXT_COL_CANDIDATES if c in df.columns), None)
if text_col is None:
    raise ValueError(f"Couldn't find a text column. Found columns: {list(df.columns)}")

texts = df[text_col].astype(str).tolist()
y_true = df["label"].astype(int).to_numpy()

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# To batch the forward passes so external evaluation stays manageable on CPU as well.
batch_size = 16
preds = []

with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=256,   
            return_tensors="pt"
        ).to(device)

        logits = model(**enc).logits
        batch_pred = torch.argmax(logits, dim=1).cpu().numpy()
        preds.append(batch_pred)

y_pred = np.concatenate(preds)

print("External dataset:", DATA_PATH)
print("Model:", MODEL_DIR)
print("Accuracy:", accuracy_score(y_true, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_true, y_pred))
print("\nReport:\n", classification_report(y_true, y_pred, digits=4))
