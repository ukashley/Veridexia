import time
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# This script is just a quick side-by-side sanity check, useful when you want a rough feel for how both saved models respond to the same message.

# Baseline
BASELINE_MODEL_PATH = "models/baseline/model.pkl"
VECTORIZER_PATH = "models/baseline/vectorizer.pkl"

baseline_model = joblib.load(BASELINE_MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Distilbert
BERT_MODEL_PATH = "models/distilbert/final_model"

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
bert_model.eval()

# Input 
email = input("\nEnter email text:\n")

print("\n" + "=" * 50)
print("PHISHING DETECTION RESULTS")
print("=" * 50)

# Baseline inference is nearly instant, so it gives a nice contrast with the heavier transformer path.
start = time.time()
X = vectorizer.transform([email])
baseline_probs = baseline_model.predict_proba(X)[0]
baseline_time = (time.time() - start) * 1000

print("\nBaseline (TF-IDF + Logistic Regression)")
print(f"  Prediction: {'PHISHING' if baseline_probs[1] > 0.5 else 'LEGITIMATE'}")
print(f"  Confidence: {baseline_probs[1]*100:.2f}%")
print(f"  Inference time: {baseline_time:.1f} ms")

# DistilBERT is slower but can pick up on context that the TF-IDF baseline misses.
start = time.time()
inputs = tokenizer(
    email,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=256
)
with torch.no_grad():
    logits = bert_model(**inputs).logits
    bert_probs = torch.softmax(logits, dim=-1)[0]
bert_time = (time.time() - start) * 1000

print("\nDistilBERT (Transformer)")
print(f"  Prediction: {'PHISHING' if bert_probs[1] > 0.5 else 'LEGITIMATE'}")
print(f"  Legitimate: {bert_probs[0]:.2f}")
print(f"  Phishing:   {bert_probs[1]:.2f}")
print(f"  Inference time: {bert_time:.1f} ms")

print("\n" + "=" * 50)
