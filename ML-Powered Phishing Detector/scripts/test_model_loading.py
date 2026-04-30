"""Smoke-test that the saved local models can be loaded and used.

Run from the project root:
    python scripts/test_model_loading.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.baseline import BaselinePredictor
from src.inference.distilbert import DistilBertPredictor


def main() -> None:
    sample = (
        "Subject: Action required\n\n"
        "Please verify your account details before the end of the day."
    )

    baseline = BaselinePredictor()
    baseline_result = baseline.predict(sample, threshold=0.65)
    print("Baseline loaded:", baseline_result)

    distilbert = DistilBertPredictor()
    distilbert_result = distilbert.predict(sample, threshold=0.65)
    print("DistilBERT loaded:", distilbert_result)


if __name__ == "__main__":
    main()
