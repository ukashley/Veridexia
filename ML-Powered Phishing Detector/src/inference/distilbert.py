from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .common import PredictionResult

class DistilBertPredictor:
    # Loads the fine-tuned transformer model used as the main classifier.
    def __init__(self, model_dir=Path("models/distilbert/final_model")):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        # The model is small enough to run on CPU for the demo, but we still take GPU if it is there.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, text: str, threshold: float = 0.5, max_length: int = 256) -> PredictionResult:
        # The training pipeline used 256 tokens, so we keep inference aligned with that setup.
        enc = self.tokenizer(text, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
            prob = torch.softmax(logits, dim=1)[0, 1].item()

        # Thresholding is done outside the model so the UI can expose sensitivity settings.
        label = int(prob >= threshold)
        return PredictionResult(label=label, prob_phishing=float(prob),
                                model_name="distilbert",
                                threshold=threshold)
