from pathlib import Path
import joblib
import numpy as np
from .common import PredictionResult

class BaselinePredictor:
    def __init__(self,
                 model_path=Path("models/baseline/model.pkl"),
                 vec_path=Path("models/baseline/vectorizer.pkl")):
        self.vectorizer = joblib.load(vec_path)
        self.model = joblib.load(model_path)

    def predict(self, text: str, threshold: float = 0.5) -> PredictionResult:
        X = self.vectorizer.transform([text])
        prob = float(self.model.predict_proba(X)[0][1])
        label = int(prob >= threshold)
        return PredictionResult(label=label, prob_phishing=prob,
                                model_name="baseline_tfidf_lr",
                                threshold=threshold)
