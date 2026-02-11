from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_PATH = Path("data/processed/trec06_processed.csv")
VEC_PATH = Path("models/baseline/vectorizer.pkl")
MODEL_PATH = Path("models/baseline/model.pkl")

df = pd.read_csv(DATA_PATH)

# Auto-detect text column
TEXT_COLS = ["text_combined", "text", "email", "body"]
text_col = next(c for c in TEXT_COLS if c in df.columns)

X = df[text_col].astype(str)
y = df["label"].astype(int)

vectorizer = joblib.load(VEC_PATH)
model = joblib.load(MODEL_PATH)

X_vec = vectorizer.transform(X)
y_pred = model.predict(X_vec)

print("Baseline External Evaluation (TREC-06)")
print("Accuracy:", accuracy_score(y, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
print("\nReport:\n", classification_report(y, y_pred, digits=4))
