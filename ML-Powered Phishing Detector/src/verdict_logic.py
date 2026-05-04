from __future__ import annotations


def compute_user_verdict(result, rules: dict, support_result=None):
    # This is the hybrid decision layer used by the app.
    # The model probability gives the starting point, then rule evidence adjusts
    # cases where normal service emails look suspicious but have reassuring context.
    prob = float(result.prob_phishing)
    threshold = float(result.threshold)
    risk_items = rules.get('signals', [])
    reassurance_items = rules.get('reassurance_signals', [])
    context_items = rules.get('context_signals', [])
    risk_keys = {item.get('key') for item in risk_items}
    reassurance_keys = {item.get('key') for item in reassurance_items}
    context_keys = {item.get('key') for item in context_items}
    risk_score = float(rules.get('risk_score', 0.0))
    support_prob = None if support_result is None else float(support_result.prob_phishing)
    risk_count = len(risk_items)

    no_strong_cues = (
        not bool(rules.get("has_credential_request"))
        and not bool(rules.get("has_threat"))
        and not bool(rules.get("has_urgency"))
        and int(rules.get("url_count", 0)) == 0
    )

    # High-risk cues that should not be ignored just because the wording looks normal.
    dangerous_risk_keys = {
        'credential_request',
        'threat_language',
        'suspicious_link',
        'domain_mismatch',
        'suspicious_sender_domain',
    }
    no_dangerous_cues = not bool(dangerous_risk_keys & risk_keys)
    has_model_support = support_prob is not None
    max_model_prob = max(prob, support_prob if support_prob is not None else 0.0)
    strong_phishing_combo = (
        (
            'credential_request' in risk_keys
            and (
                bool({'urgency', 'threat_language', 'suspicious_link', 'domain_mismatch', 'suspicious_sender_domain'} & risk_keys)
                or max_model_prob >= 0.55
            )
        )
        or (
            'suspicious_link' in risk_keys
            and (
                bool({'domain_mismatch', 'suspicious_sender_domain'} & risk_keys)
                or max_model_prob >= threshold
            )
        )
        or (
            risk_score >= 3.0
            and max_model_prob >= 0.45
        )
    )

    near_boundary = abs(prob - threshold) < 0.12
    review_recommended = near_boundary or (prob >= threshold and (no_strong_cues or no_dangerous_cues))
    benign_context = (
        'sender_email_present' in context_keys
        and 'email_address_present' in context_keys
        and 'domain_mismatch' not in risk_keys
        and 'suspicious_link' not in risk_keys
    )
    clean_message = no_strong_cues and risk_score <= 0 and 'domain_mismatch' not in risk_keys and 'suspicious_link' not in risk_keys
    low_severity_risk_keys = {'payment_language', 'generic_greeting'}
    low_severity_only = bool(risk_keys) and risk_keys.issubset(low_severity_risk_keys)

    # Routine categories that often caused false positives during manual testing.
    routine_message_keys = {
        'transactional_notification',
        'account_administration_notice',
        'newsletter_context',
        'marketing_promotion_context',
        'job_alert_context',
        'identity_verification_context',
        'security_notification',
        'account_access_notification',
        'formal_service_message',
    }
    trusted_service_keys = {
        'sender_brand_match',
        'domain_match',
        'link_domain_match',
        'brand_related_link',
        'official_website_present',
        'support_contact_present',
        'no_email_credential_request',
        'security_notification',
        'in_app_security_instruction',
    }
    transactional_reassurance = bool(
        (routine_message_keys | trusted_service_keys) & reassurance_keys
    )
    strong_benign_reassurance = bool(
        routine_message_keys & reassurance_keys
    ) and bool({'sender_brand_match', 'domain_match', 'link_domain_match'} & reassurance_keys)
    trusted_service_context = (
        no_dangerous_cues
        and 'domain_mismatch' not in risk_keys
        and 'suspicious_link' not in risk_keys
        and bool({'sender_brand_match', 'domain_match', 'link_domain_match'} & reassurance_keys)
        and bool({'transactional_notification', 'formal_service_message', 'support_contact_present', 'official_website_present'} & reassurance_keys)
        and bool({'no_email_credential_request', 'security_notification'} & reassurance_keys)
        and risk_score <= 1.2
    )
    safe_routine_message = (
        bool(routine_message_keys & reassurance_keys)
        and no_dangerous_cues
        and 'domain_mismatch' not in risk_keys
        and 'suspicious_link' not in risk_keys
        and risk_score <= 0
    )
    clear_marketing_newsletter = (
        {'newsletter_context', 'marketing_promotion_context'} <= reassurance_keys
        and bool({'sender_brand_match', 'domain_match', 'link_domain_match', 'brand_related_link'} & reassurance_keys)
        and no_dangerous_cues
        and 'domain_mismatch' not in risk_keys
        and 'suspicious_link' not in risk_keys
        and risk_score <= 0.5
    )
    account_security_notice = (
        {'security_notification', 'in_app_security_instruction'} <= reassurance_keys
        and bool({'sender_brand_match', 'domain_match', 'link_domain_match', 'brand_related_link', 'no_email_credential_request'} & reassurance_keys)
        and 'credential_request' not in risk_keys
        and 'domain_mismatch' not in risk_keys
        and 'suspicious_link' not in risk_keys
        and 'suspicious_sender_domain' not in risk_keys
        and risk_score <= 0.5
    )
    account_access_notice = (
        'account_access_notification' in reassurance_keys
        and no_dangerous_cues
        and 'credential_request' not in risk_keys
        and 'domain_mismatch' not in risk_keys
        and 'suspicious_link' not in risk_keys
        and 'suspicious_sender_domain' not in risk_keys
        and 'sender_email_present' in context_keys
        and risk_score <= 0
    )
    benign_model_disagreement = (
        prob >= threshold
        and has_model_support
        and support_prob <= 0.45
        and no_dangerous_cues
        and risk_score <= 0.5
        and (
            bool(routine_message_keys & reassurance_keys)
            or bool({'sender_brand_match', 'domain_match', 'link_domain_match', 'security_notification', 'identity_verification_context'} & reassurance_keys)
            or benign_context
            or not risk_keys
        )
    )
    display_prob = prob
    decision_basis = 'model'

    # Manual overrides keep the displayed verdict aligned with the evidence layer.
    # They do not retrain the model; they only make the final user-facing result safer to interpret.
    if strong_phishing_combo:
        main_label = "Phishing"
        level = "high" if max_model_prob >= 0.75 or risk_score >= 3.4 else "medium"
        display_prob = max(max_model_prob, threshold + 0.05)
        decision_basis = 'high_risk_evidence_override'
    elif (
        prob >= threshold
        and 'identity_verification_context' in reassurance_keys
        and ('sender_email_present' in context_keys or bool({'sender_brand_match', 'domain_match'} & reassurance_keys))
        and no_dangerous_cues
        and 'domain_mismatch' not in risk_keys
        and 'suspicious_link' not in risk_keys
    ):
        main_label = "Legitimate"
        level = "review"
        display_prob = min(
            support_prob if support_prob is not None else threshold - 0.01,
            max(0.0, threshold - 0.01),
        )
        review_recommended = False
        decision_basis = 'identity_verification_override'
    elif (
        prob >= threshold
        and trusted_service_context
        and (support_prob is None or support_prob <= 0.90)
    ):
        main_label = "Legitimate"
        level = "review"
        display_prob = min(
            support_prob if support_prob is not None else threshold - 0.01,
            max(0.0, threshold - 0.01),
        )
        review_recommended = True
        decision_basis = 'trusted_service_context_override'
    elif (
        prob >= threshold
        and clear_marketing_newsletter
    ):
        main_label = "Legitimate"
        level = "review"
        display_prob = max(0.0, threshold - 0.01)
        review_recommended = True
        decision_basis = 'marketing_newsletter_override'
    elif (
        prob >= threshold
        and account_security_notice
    ):
        main_label = "Legitimate"
        level = "review"
        display_prob = max(0.0, threshold - 0.01)
        review_recommended = True
        decision_basis = 'account_security_notice_override'
    elif (
        prob >= threshold
        and account_access_notice
    ):
        main_label = "Legitimate"
        level = "review"
        display_prob = max(0.0, threshold - 0.01)
        review_recommended = True
        decision_basis = 'account_access_notification_override'
    elif (
        prob >= threshold
        and safe_routine_message
        and bool({'sender_brand_match', 'domain_match', 'link_domain_match'} & reassurance_keys)
        and (support_prob is None or support_prob <= 0.85)
    ):
        main_label = "Legitimate"
        level = "review"
        display_prob = min(
            support_prob if support_prob is not None else threshold - 0.01,
            max(0.0, threshold - 0.01),
        )
        decision_basis = 'benign_routine_override'
    elif benign_model_disagreement:
        main_label = "Legitimate"
        level = "review"
        display_prob = min(support_prob, max(0.0, threshold - 0.01))
        decision_basis = 'benign_support_model_override'
    elif prob >= threshold and near_boundary and no_dangerous_cues and risk_score <= 0 and (
        benign_context or bool({'domain_match', 'link_domain_match'} & reassurance_keys)
    ):
        main_label = "Legitimate"
        level = "review"
        display_prob = max(0.0, threshold - 0.01)
        decision_basis = 'hybrid_override'
    elif (
        prob >= threshold
        and support_prob is not None
        and support_prob < threshold
        and no_dangerous_cues
        and low_severity_only
        and risk_count <= 1
        and transactional_reassurance
        and 'domain_mismatch' not in risk_keys
        and 'suspicious_link' not in risk_keys
    ):
        main_label = "Legitimate"
        level = "review"
        display_prob = support_prob
        decision_basis = 'transactional_override'
    elif (
        prob >= threshold
        and support_prob is not None
        and (clean_message or safe_routine_message)
        and ('sender_email_present' in context_keys or benign_context or bool({'domain_match', 'link_domain_match'} & reassurance_keys))
        and support_prob <= 0.40
    ):
        ensemble_prob = round((prob + support_prob) / 2.0, 4)
        if ensemble_prob < threshold:
            main_label = "Legitimate"
            level = "review"
            display_prob = ensemble_prob
            decision_basis = 'cross_model_override'
        else:
            main_label = "Phishing"
            level = "medium"
    elif (
        prob >= threshold
        and no_dangerous_cues
        and risk_score <= -1.0
        and strong_benign_reassurance
        and 'domain_mismatch' not in risk_keys
        and 'suspicious_link' not in risk_keys
    ):
        main_label = "Legitimate"
        level = "review"
        display_prob = max(0.0, threshold - 0.01)
        decision_basis = 'benign_notice_override'
    elif (
        prob >= threshold
        and support_prob is not None
        and no_dangerous_cues
        and risk_score <= -0.6
        and bool(routine_message_keys & reassurance_keys)
        and support_prob <= max(0.55, threshold - 0.05)
        and 'domain_mismatch' not in risk_keys
        and 'suspicious_link' not in risk_keys
    ):
        main_label = "Legitimate"
        level = "review"
        display_prob = min(support_prob, max(0.0, threshold - 0.01))
        decision_basis = 'routine_message_override'
    elif prob >= threshold:
        main_label = "Phishing"
        level = "high" if prob >= max(0.80, threshold + 0.20) else "medium"
    elif (
        has_model_support
        and support_prob >= threshold
        and not safe_routine_message
        and (
            bool(dangerous_risk_keys & risk_keys)
            or 'urgency' in risk_keys
            or risk_score >= 1.4
        )
    ):
        main_label = "Phishing"
        level = "medium"
        display_prob = support_prob
        decision_basis = 'support_model_risk_override'
    else:
        main_label = "Legitimate"
        level = "review" if risk_score > 0 and near_boundary else "low"

    return {
        "label": main_label,
        "level": level,
        "review_recommended": review_recommended,
        "display_prob": display_prob,
        "model_prob": prob,
        "decision_basis": decision_basis,
        "support_prob": support_prob,
    }


def get_risk_level(label: str, probability: float, threshold: float, review_recommended: bool = False) -> str:
    prob = float(probability)
    boundary_gap = abs(prob - float(threshold))

    if label == 'Phishing':
        if prob >= 0.80 or boundary_gap >= 0.18:
            return 'High'
        return 'Medium'

    if review_recommended:
        return 'Medium'
    if prob <= 0.25:
        return 'Very low'
    return 'Low'


def result_advice(label: str, risk_level: str, probability: float, threshold: float) -> str:
    if risk_level == 'Medium' and abs(float(probability) - float(threshold)) < 0.08:
        return 'Borderline result. Review the message manually before clicking links or sharing details.'
    if label == 'Phishing':
        if risk_level == 'High':
            return 'Do not click links or provide credentials. Verify through the official website.'
        return 'High risk. Treat this message as suspicious.'
    return 'Low risk, but review links before clicking.'
