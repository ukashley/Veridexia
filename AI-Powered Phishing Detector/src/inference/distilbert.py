import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model configuration
MODEL_PATH = Path("models/distilbert/final_model")

# To load tokenizer only once at import time
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  

# Public API
def predict(text: str) -> dict:
    """
    Run phishing detection on a single email.

    Args:
        text (str): Raw email text

    Returns:
        dict: prediction result (label, probabilities, inference time)
    """
    # You will implement this step by step
    raise NotImplementedError("predict() not implemented yet")
