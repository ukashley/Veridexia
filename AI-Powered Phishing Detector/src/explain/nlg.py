def generate_explanation(result, evidence=None, rule_evidence=None):
    label_str = "PHISHING" if result.label == 1 else "LEGITIMATE"
    prob = float(result.prob_phishing)

    parts = []
    parts.append(
        f"The system classified this email as {label_str} with an estimated phishing probability of {prob:.2f} "
        f"(threshold = {float(result.threshold):.2f})."
    )

    # Rule-based cues for both models
    if rule_evidence:
        signals = rule_evidence.get("signals", [])
        url_count = rule_evidence.get("url_count", 0)

        if signals:
            pretty = ", ".join(signals).replace("_", " ")
            parts.append(f"Risk indicators detected: {pretty}.")
        else:
            parts.append("No strong phishing indicators (e.g., urgency, threats, credential prompts) were detected in the text.")

        if url_count and url_count > 0:
            parts.append(f"The email contains {url_count} link(s), which can be used for redirection or credential harvesting.")
        else:
            parts.append("No links were detected in the message.")

    # Baseline evidence 
    if evidence and isinstance(evidence, dict) and evidence.get("top_terms"):
        top_terms = [t for t, _ in evidence["top_terms"][:6]]
        parts.append(f"Influential terms (baseline): {', '.join(top_terms)}.")

    # Warning if cues conflict with model decision
    if rule_evidence and rule_evidence.get("has_password_request") and result.label == 0:
        parts.append(
            "Warning: This message contains a credential request (e.g., asking for a password/OTP), which is a high-risk phishing pattern. "
            "Even if the model probability is below the threshold, treat this message with caution."
        )

    return " ".join(parts)
