import numpy as np

def baseline_evidence(text: str, predictor, top_k: int = 8):
    vec = predictor.vectorizer
    clf = predictor.model
    X = vec.transform([text])
    feature_names = np.array(vec.get_feature_names_out())

    # Logistic regression weights for phishing class
    weights = clf.coef_[0]
    present_idx = X.nonzero()[1]

    scored = [(feature_names[i], float(weights[i])) for i in present_idx]
    scored.sort(key=lambda x: x[1], reverse=True)

    return {"top_terms": scored[:top_k]}
