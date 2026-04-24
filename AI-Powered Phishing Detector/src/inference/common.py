from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class PredictionResult:
    # This object is the shared handoff format between the predictors,
    # the rule layer, and the explanation UI.
    label: int                 # 0 legit, 1 phishing
    prob_phishing: float       # 0.0 - 1.0
    model_name: str
    threshold: float
    evidence: Optional[Dict] = None  # filled later by evidence module
