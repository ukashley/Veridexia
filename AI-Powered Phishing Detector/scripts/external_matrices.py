import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "external"

BASELINE_JSON = RESULTS_DIR / "trec06_baseline_metrics.json"
DISTILBERT_JSON = RESULTS_DIR / "trec06_distilbert_metrics.json"

OUT_BASELINE = RESULTS_DIR / "trec06_baseline_confusion_matrix.png"
OUT_DISTILBERT = RESULTS_DIR / "trec06_distilbert_confusion_matrix.png"


def generate_cm(json_path: Path, out_path: Path, title: str):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    cm = np.array(data["confusion_matrix"])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not BASELINE_JSON.exists():
        raise FileNotFoundError(f"Missing: {BASELINE_JSON}")
    if not DISTILBERT_JSON.exists():
        raise FileNotFoundError(f"Missing: {DISTILBERT_JSON}")

    generate_cm(BASELINE_JSON, OUT_BASELINE, "Confusion Matrix - Baseline (TREC-06)")
    generate_cm(DISTILBERT_JSON, OUT_DISTILBERT, "Confusion Matrix - DistilBERT (TREC-06)")
