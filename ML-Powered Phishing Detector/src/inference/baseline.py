from pathlib import Path
import joblib
import numpy as np
from .common import PredictionResult

class BaselinePredictor:
    # Loads the saved TF-IDF vectorizer and Logistic Regression model used as
    # the lightweight comparison model.
    def __init__(self,
                 model_path=Path("models/baseline/model.pkl"),
                 vec_path=Path("models/baseline/vectorizer.pkl")):
        # The baseline is intentionally simple to keep loading fast in the app
        # and to give us a transparent comparison point against DistilBERT.
        self.vectorizer = joblib.load(vec_path)
        self.model = joblib.load(model_path)

    def predict(self, text: str, threshold: float = 0.5) -> PredictionResult:
        # Convert the email into TF-IDF features, then read the phishing class probability.
        X = self.vectorizer.transform([text])
        prob = float(self.model.predict_proba(X)[0][1])
        label = int(prob >= threshold)
        # We keep the threshold outside the saved model so the UI can adjust
        # sensitivity without retraining anything.
        return PredictionResult(label=label, prob_phishing=prob,
                                model_name="baseline_tfidf_lr",
                                threshold=threshold)
