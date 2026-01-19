"""
Trains a classical ML baseline (TF-IDF + Logistic Regression) for comparison
with the DistilBERT model. This demonstrates understanding of model evaluation
and provides a performance benchmark.
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Paths
PROC_DIR = Path("data/processed")
MODEL_DIR = Path("models/baseline")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("BASELINE MODEL TRAINING - TF-IDF + LOGISTIC REGRESSION")
print("="*70)

# To Load Processed Data
print("\n[1] Loading processed data...")
train_df = pd.read_csv(PROC_DIR / "train.csv")
val_df = pd.read_csv(PROC_DIR / "val.csv")
test_df = pd.read_csv(PROC_DIR / "test.csv")

print(f"✓ Data loaded:")
print(f"  • Train: {len(train_df):,} samples")
print(f"  • Val:   {len(val_df):,} samples")
print(f"  • Test:  {len(test_df):,} samples")

# Combining train and val for final training 
train_full_df = pd.concat([train_df, val_df], ignore_index=True)
print(f"  • Combined train+val: {len(train_full_df):,} samples")

# Class distribution testing
print(f"\n  Class distribution (test set):")
print(f"  • Legitimate: {(test_df['label']==0).sum():,} ({(test_df['label']==0).mean():.1%})")
print(f"  • Phishing:   {(test_df['label']==1).sum():,} ({(test_df['label']==1).mean():.1%})")

# TF-IDF VECTORIZATION
print("\n[2] Creating TF-IDF features...")
print("  (This extracts word frequency features from text)")

start_time = time.time()

# Initializing TF-IDF with reasonable parameters
tfidf = TfidfVectorizer(
    max_features=10000,      # Top 10k most important words
    ngram_range=(1, 2),      # Unigrams and bigrams
    min_df=2,                # Ignore the very rare words
    max_df=0.95,             # Ignore the very common words
    strip_accents='unicode',
    lowercase=True,
    stop_words='english'     # Remove common English stopwords
)

# Fit on training data and transform all sets
X_train = tfidf.fit_transform(train_full_df['text'])
X_test = tfidf.transform(test_df['text'])

y_train = train_full_df['label'].values
y_test = test_df['label'].values

vectorize_time = time.time() - start_time

print(f"✓ TF-IDF vectorization complete ({vectorize_time:.2f}s)")
print(f"  • Vocabulary size: {len(tfidf.vocabulary_):,} words")
print(f"  • Feature matrix shape: {X_train.shape}")
print(f"  • Sparsity: {(1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1])):.2%}")

# Show top features
feature_names = tfidf.get_feature_names_out()
print(f"\n  Sample features (top 20):")
print(f"  {', '.join(feature_names[:20])}")

# TRAIN LOGISTIC REGRESSION 
print("\n[3] Training Logistic Regression...")
print("  (Using class weights to handle imbalance)")

start_time = time.time()

# TrainING with class weights to handle imbalance
clf = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  
    random_state=42,
    solver='lbfgs',
    n_jobs=-1,               # Use all CPU cores
    verbose=0
)

clf.fit(X_train, y_train)
train_time = time.time() - start_time

print(f"✓ Training complete ({train_time:.2f}s)")
print(f"  • Converged: {clf.n_iter_[0] < clf.max_iter}")
print(f"  • Iterations: {clf.n_iter_[0]}")

# EVALUATE ON TEST SET 
print("\n[4] Evaluating on test set...")

# Predictions
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# To calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average='binary', pos_label=1
)

# Per-class metrics
precision_pc, recall_pc, f1_pc, _ = precision_recall_fscore_support(
    y_test, y_pred, average=None
)

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n✓ Test Results:")
print(f"  • Accuracy:  {accuracy:.4f} ({accuracy:.2%})")
print(f"  • Precision: {precision:.4f} ({precision:.2%})")
print(f"  • Recall:    {recall:.4f} ({recall:.2%})")
print(f"  • F1-score:  {f1:.4f} ({f1:.2%})")
print(f"  • ROC-AUC:   {roc_auc:.4f}")

print(f"\n  Per-class metrics:")
print(f"  • Legitimate - Precision: {precision_pc[0]:.4f}, Recall: {recall_pc[0]:.4f}, F1: {f1_pc[0]:.4f}")
print(f"  • Phishing   - Precision: {precision_pc[1]:.4f}, Recall: {recall_pc[1]:.4f}, F1: {f1_pc[1]:.4f}")

# CONFUSION MATRIX 
print("\n[5] Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=['Legitimate', 'Phishing'],
    yticklabels=['Legitimate', 'Phishing'],
    cbar_kws={'label': 'Count'},
    ax=ax
)
ax.set_title('Confusion Matrix - Baseline (TF-IDF + LogReg)', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontweight='bold')
ax.set_xlabel('Predicted Label', fontweight='bold')
plt.tight_layout()
plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=300, bbox_inches='tight')
print(f"✓ Confusion matrix saved to {MODEL_DIR / 'confusion_matrix.png'}")

# ROC CURVE 
print("\n[6] Generating ROC curve...")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='#e74c3c', linewidth=2, label=f'Baseline (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontweight='bold')
ax.set_ylabel('True Positive Rate', fontweight='bold')
ax.set_title('ROC Curve - Baseline Model', fontsize=14, fontweight='bold')
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(MODEL_DIR / "roc_curve.png", dpi=300, bbox_inches='tight')
print(f"✓ ROC curve saved to {MODEL_DIR / 'roc_curve.png'}")

# FEATURE IMPORTANCE 
print("\n[7] Analyzing feature importance...")

# Get the top features for each class
coefficients = clf.coef_[0]
top_phishing_idx = np.argsort(coefficients)[-20:][::-1]
top_legit_idx = np.argsort(coefficients)[:20]

top_phishing_features = [(feature_names[i], coefficients[i]) for i in top_phishing_idx]
top_legit_features = [(feature_names[i], coefficients[i]) for i in top_legit_idx]

print(f"\n  Top 10 features indicating PHISHING:")
for word, coef in top_phishing_features[:10]:
    print(f"    {word:<30} {coef:>8.4f}")

print(f"\n  Top 10 features indicating LEGITIMATE:")
for word, coef in top_legit_features[:10]:
    print(f"    {word:<30} {coef:>8.4f}")

# Visualize feature importance
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Phishing indicators
words_p, coefs_p = zip(*top_phishing_features[:15])
axes[0].barh(range(len(words_p)), coefs_p, color='#e74c3c', alpha=0.8, edgecolor='black')
axes[0].set_yticks(range(len(words_p)))
axes[0].set_yticklabels(words_p)
axes[0].invert_yaxis()
axes[0].set_xlabel('Coefficient (→ Phishing)', fontweight='bold')
axes[0].set_title('Top Features for Phishing Detection', fontweight='bold', fontsize=12)

# Legitimate indicators
words_l, coefs_l = zip(*top_legit_features[:15])
axes[1].barh(range(len(words_l)), coefs_l, color='#2ecc71', alpha=0.8, edgecolor='black')
axes[1].set_yticks(range(len(words_l)))
axes[1].set_yticklabels(words_l)
axes[1].invert_yaxis()
axes[1].set_xlabel('Coefficient (→ Legitimate)', fontweight='bold')
axes[1].set_title('Top Features for Legitimate Detection', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(MODEL_DIR / "feature_importance.png", dpi=300, bbox_inches='tight')
print(f"✓ Feature importance saved to {MODEL_DIR / 'feature_importance.png'}")

# SAVE MODEL AND METRICS 
print("\n[8] Saving model and metrics...")

# Save model
with open(MODEL_DIR / "model.pkl", "wb") as f:
    pickle.dump(clf, f)

# Save vectorizer
with open(MODEL_DIR / "vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# Save metrics
metrics = {
    "model": "TF-IDF + Logistic Regression",
    "training": {
        "train_samples": len(train_full_df),
        "test_samples": len(test_df),
        "vectorization_time": float(vectorize_time),
        "training_time": float(train_time),
        "total_time": float(vectorize_time + train_time),
        "max_features": 10000,
        "ngram_range": [1, 2],
        "vocabulary_size": len(tfidf.vocabulary_)
    },
    "test": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "precision_legitimate": float(precision_pc[0]),
        "recall_legitimate": float(recall_pc[0]),
        "f1_legitimate": float(f1_pc[0]),
        "precision_phishing": float(precision_pc[1]),
        "recall_phishing": float(recall_pc[1]),
        "f1_phishing": float(f1_pc[1])
    },
    "confusion_matrix": cm.tolist(),
    "top_phishing_features": top_phishing_features[:20],
    "top_legitimate_features": top_legit_features[:20]
}

with open(MODEL_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✓ Model saved to {MODEL_DIR / 'model.pkl'}")
print(f"✓ Vectorizer saved to {MODEL_DIR / 'vectorizer.pkl'}")
print(f"✓ Metrics saved to {MODEL_DIR / 'metrics.json'}")

# SUMMARY
print("\n" + "="*70)
print("BASELINE TRAINING COMPLETE!")
print("="*70)

print(f"\n Final Test Results:")
print(f"  • Accuracy:  {accuracy:.2%}")
print(f"  • Precision: {precision:.2%}")
print(f"  • Recall:    {recall:.2%}")
print(f"  • F1-score:  {f1:.2%}")
print(f"  • ROC-AUC:   {roc_auc:.4f}")

print(f"\n Performance:")
print(f"  • Vectorization: {vectorize_time:.2f}s")
print(f"  • Training: {train_time:.2f}s")
print(f"  • Total: {vectorize_time + train_time:.2f}s")

print(f"\n Outputs:")
print(f"  • Model: {MODEL_DIR / 'model.pkl'}")
print(f"  • Vectorizer: {MODEL_DIR / 'vectorizer.pkl'}")
print(f"  • Metrics: {MODEL_DIR / 'metrics.json'}")
print(f"  • Confusion matrix: {MODEL_DIR / 'confusion_matrix.png'}")
print(f"  • ROC curve: {MODEL_DIR / 'roc_curve.png'}")
print(f"  • Feature importance: {MODEL_DIR / 'feature_importance.png'}")

print("\n Ready for model comparison!")