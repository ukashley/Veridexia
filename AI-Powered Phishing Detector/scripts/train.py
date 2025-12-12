"""
train.py:
Fine-tunes DistilBERT for phishing email detection.
Handles class imbalance with weighted loss, saves best model, and logs metrics.
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    # Paths
    TOK_DIR = Path("data/processed/tokenized")
MODEL_DIR = Path("models/distilbert")
STATS_FILE = Path("data/processed/dataset_stats.json")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("DISTILBERT PHISHING DETECTOR - TRAINING")
print("="*70)

# ===== LOAD DATASET STATISTICS =====
print("\n[1] Loading dataset statistics...")
with open(STATS_FILE, "r") as f:
    stats = json.load(f)

print(f"✓ Dataset info:")
print(f"  • Train: {stats['dataset_info']['train_samples']:,} samples")
print(f"  • Val:   {stats['dataset_info']['val_samples']:,} samples")
print(f"  • Test:  {stats['dataset_info']['test_samples']:,} samples")
print(f"  • Class distribution: {stats['class_distribution']['legitimate_ratio']:.1%} legitimate, {stats['class_distribution']['phishing_ratio']:.1%} phishing")

# ===== LOAD TOKENIZED DATASETS =====
print("\n[2] Loading tokenized datasets...")
train_dataset = load_from_disk(str(TOK_DIR / "train"))
val_dataset = load_from_disk(str(TOK_DIR / "val"))
test_dataset = load_from_disk(str(TOK_DIR / "test"))

print(f"✓ Loaded datasets:")
print(f"  • Train: {len(train_dataset):,} samples")
print(f"  • Val:   {len(val_dataset):,} samples")
print(f"  • Test:  {len(test_dataset):,} samples")

# ===== CALCULATE CLASS WEIGHTS =====
print("\n[3] Calculating class weights for imbalanced data...")

# Count labels in training set
train_labels = train_dataset["label"]
label_counts = np.bincount(train_labels)
total_samples = len(train_labels)

# Calculate weights (inverse frequency)
class_weights = total_samples / (len(label_counts) * label_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

print(f"✓ Class weights calculated:")
print(f"  • Legitimate (0): {class_weights[0]:.3f} (count: {label_counts[0]:,})")
print(f"  • Phishing (1):   {class_weights[1]:.3f} (count: {label_counts[1]:,})")
print(f"  → This helps the model pay more attention to the minority class")

# ===== LOAD MODEL =====
print("\n[4] Loading DistilBERT model...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Custom model class with weighted loss
class WeightedDistilBERT(torch.nn.Module):
    def __init__(self, model_name, class_weights):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,
            problem_type="single_label_classification"
        )
        self.class_weights = class_weights
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        # Apply weighted loss if labels provided
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
            loss = loss_fct(outputs.logits, labels)
            return {"loss": loss, "logits": outputs.logits}
        
        return outputs

model = WeightedDistilBERT(model_name, class_weights_tensor)
print(f"✓ Model loaded: {model_name}")
print(f"  • Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  • Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ===== DEFINE METRICS =====
def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', pos_label=1
    )
    accuracy = accuracy_score(labels, predictions)
    
    # Calculate per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_legitimate': precision_per_class[0],
        'recall_legitimate': recall_per_class[0],
        'f1_legitimate': f1_per_class[0],
        'precision_phishing': precision_per_class[1],
        'recall_phishing': recall_per_class[1],
        'f1_phishing': f1_per_class[1],
    }

# ===== TRAINING ARGUMENTS =====
print("\n[5] Setting up training configuration...")

training_args = TrainingArguments(
    output_dir=str(MODEL_DIR / "checkpoints"),
    
    # Training hyperparameters
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    
    # Evaluation strategy
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    
    # Logging
    logging_dir=str(MODEL_DIR / "logs"),
    logging_steps=50,
    report_to="none",  # Disable wandb/tensorboard
    
    # Performance
    fp16=torch.cuda.is_available(),  # Mixed precision if GPU available
    dataloader_num_workers=0,  # Changed from 2 to 0 for Windows compatibility
    
    # Reproducibility
    seed=42,
)

print(f"✓ Training configuration:")
print(f"  • Epochs: {training_args.num_train_epochs}")
print(f"  • Batch size: {training_args.per_device_train_batch_size}")
print(f"  • Learning rate: {training_args.learning_rate}")
print(f"  • Device: {'GPU (fp16)' if training_args.fp16 else 'CPU'}")

# ===== INITIALIZE TRAINER =====
print("\n[6] Initializing trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("✓ Trainer initialized with early stopping (patience=3)")

# ===== TRAIN MODEL =====
print("\n[7] Starting training...")
print("-"*70)

train_result = trainer.train()

print("-"*70)
print("✓ Training complete!")
print(f"  • Training time: {train_result.metrics['train_runtime']:.2f}s")
print(f"  • Training loss: {train_result.metrics['train_loss']:.4f}")

# ===== EVALUATE ON VALIDATION SET =====
print("\n[8] Evaluating on validation set...")
val_metrics = trainer.evaluate(eval_dataset=val_dataset)

print(f"✓ Validation metrics:")
print(f"  • Accuracy:  {val_metrics['eval_accuracy']:.4f}")
print(f"  • Precision: {val_metrics['eval_precision']:.4f}")
print(f"  • Recall:    {val_metrics['eval_recall']:.4f}")
print(f"  • F1-score:  {val_metrics['eval_f1']:.4f}")
print(f"\n  Per-class (Phishing detection):")
print(f"  • Precision: {val_metrics['eval_precision_phishing']:.4f}")
print(f"  • Recall:    {val_metrics['eval_recall_phishing']:.4f}")
print(f"  • F1-score:  {val_metrics['eval_f1_phishing']:.4f}")

# ===== EVALUATE ON TEST SET =====
print("\n[9] Evaluating on test set...")
test_metrics = trainer.evaluate(eval_dataset=test_dataset)

print(f"✓ Test metrics:")
print(f"  • Accuracy:  {test_metrics['eval_accuracy']:.4f}")
print(f"  • Precision: {test_metrics['eval_precision']:.4f}")
print(f"  • Recall:    {test_metrics['eval_recall']:.4f}")
print(f"  • F1-score:  {test_metrics['eval_f1']:.4f}")
print(f"\n  Per-class (Phishing detection):")
print(f"  • Precision: {test_metrics['eval_precision_phishing']:.4f}")
print(f"  • Recall:    {test_metrics['eval_recall_phishing']:.4f}")
print(f"  • F1-score:  {test_metrics['eval_f1_phishing']:.4f}")

# ===== GENERATE CONFUSION MATRIX =====
print("\n[10] Generating confusion matrix...")
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=-1)
y_true = predictions.label_ids

cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing'],
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - DistilBERT', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontweight='bold')
plt.xlabel('Predicted Label', fontweight='bold')
plt.tight_layout()
plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=300, bbox_inches='tight')
print(f"✓ Confusion matrix saved to {MODEL_DIR / 'confusion_matrix.png'}")

# ===== SAVE MODEL =====
print("\n[11] Saving model and metrics...")

# Save the base model (not the wrapper)
trainer.model.model.save_pretrained(MODEL_DIR / "final_model")
tokenizer.save_pretrained(MODEL_DIR / "final_model") 

# Save metrics
all_metrics = {
    "model": "distilbert-base-uncased",
    "training": {
        "epochs": training_args.num_train_epochs,
        "batch_size": training_args.per_device_train_batch_size,
        "learning_rate": training_args.learning_rate,
        "train_loss": float(train_result.metrics['train_loss']),
        "train_runtime": float(train_result.metrics['train_runtime'])
    },
    "validation": {
        "accuracy": float(val_metrics['eval_accuracy']),
        "precision": float(val_metrics['eval_precision']),
        "recall": float(val_metrics['eval_recall']),
        "f1": float(val_metrics['eval_f1']),
        "precision_phishing": float(val_metrics['eval_precision_phishing']),
        "recall_phishing": float(val_metrics['eval_recall_phishing']),
        "f1_phishing": float(val_metrics['eval_f1_phishing'])
    },
    "test": {
        "accuracy": float(test_metrics['eval_accuracy']),
        "precision": float(test_metrics['eval_precision']),
        "recall": float(test_metrics['eval_recall']),
        "f1": float(test_metrics['eval_f1']),
        "precision_phishing": float(test_metrics['eval_precision_phishing']),
        "recall_phishing": float(test_metrics['eval_recall_phishing']),
        "f1_phishing": float(test_metrics['eval_f1_phishing'])
    },
    "confusion_matrix": cm.tolist(),
    "class_weights": class_weights.tolist()
}

with open(MODEL_DIR / "metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

print(f"✓ Model saved to {MODEL_DIR / 'final_model'}")
print(f"✓ Metrics saved to {MODEL_DIR / 'metrics.json'}")

# ===== SUMMARY =====
print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\n Final Test Results:")
print(f"  • Accuracy:  {test_metrics['eval_accuracy']:.2%}")
print(f"  • Precision: {test_metrics['eval_precision']:.2%}")
print(f"  • Recall:    {test_metrics['eval_recall']:.2%}")
print(f"  • F1-score:  {test_metrics['eval_f1']:.2%}")

print(f"\n Phishing Detection Performance:")
print(f"  • Precision: {test_metrics['eval_precision_phishing']:.2%} (of flagged emails, % actually phishing)")
print(f"  • Recall:    {test_metrics['eval_recall_phishing']:.2%} (% of phishing emails caught)")
print(f"  • F1-score:  {test_metrics['eval_f1_phishing']:.2%} (harmonic mean)")

print(f"\n Outputs:")
print(f"  • Model: {MODEL_DIR / 'final_model'}")
print(f"  • Metrics: {MODEL_DIR / 'metrics.json'}")
print(f"  • Confusion matrix: {MODEL_DIR / 'confusion_matrix.png'}")

print("\n Ready for baseline comparison and deployment!")
