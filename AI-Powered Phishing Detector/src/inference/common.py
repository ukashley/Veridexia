from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class PredictionResult:
    label: int                 # 0 legit, 1 phishing
    prob_phishing: float       # 0.0 - 1.0
    model_name: str
    threshold: float
    evidence: Optional[Dict] = None  # filled later by evidence module
