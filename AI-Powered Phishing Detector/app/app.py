import sys
from pathlib import Path
import json
import time

import streamlit as st
import pandas as pd

# Project Root

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.baseline import BaselinePredictor
from src.inference.distilbert import DistilBertPredictor
from src.explain.baseline_evidence import baseline_evidence
from src.explain.nlg import generate_explanation
from src.explain.rule_evidence import rule_based_evidence

# Paths
DATASET_STATS = ROOT / "data" / "processed" / "dataset_stats.json"

BASELINE_DIR = ROOT / "models" / "baseline"
DISTILBERT_DIR = ROOT / "models" / "distilbert"
RESULTS_DIR = ROOT / "results" / "external"

BASELINE_METRICS = RESULTS_DIR / "trec06_baseline_metrics.json"
DISTILBERT_METRICS = RESULTS_DIR / "trec06_distilbert_metrics.json"

# External confusion matrices from external validation metrics 
BASELINE_CM_IMG = RESULTS_DIR / "trec06_baseline_confusion_matrix.png"
DISTILBERT_CM_IMG = RESULTS_DIR / "trec06_distilbert_confusion_matrix.png"

# Hidden in UI screen 
BASELINE_ROC_IMG = BASELINE_DIR / "roc_curve.png"
BASELINE_FEAT_IMG = BASELINE_DIR / "feature_importance.png"

# Helpers
def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def fmt_pct(x):
    try:
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "—"

def get_external_kpis(m: dict):
    """
    Extract KPIs from external metrics JSON of the form:
    { accuracy, confusion_matrix, report: { '0': {...}, '1': {...}, 'macro avg': {...}, ... } }
    """
    if not m or "report" not in m:
        return None

    rep = m["report"]
    cls1 = rep.get("1", {})  # phishing/spam mapped class
    cls0 = rep.get("0", {})  # legitimate
    macro = rep.get("macro avg", {})
    weighted = rep.get("weighted avg", {})

    return {
        "accuracy": m.get("accuracy"),
        "f1_macro": macro.get("f1-score"),
        "f1_weighted": weighted.get("f1-score"),
        "recall_phish": cls1.get("recall"),
        "precision_phish": cls1.get("precision"),
        "f1_phish": cls1.get("f1-score"),
        "recall_legit": cls0.get("recall"),
    }

def verdict_label(is_phishing: bool):
    return "PHISHING" if is_phishing else "LEGITIMATE"

def severity_color(p_phish: float, threshold: float):
    if p_phish >= max(threshold, 0.8):
        return "🔴 High risk"
    if p_phish >= threshold:
        return "🟠 Medium risk"
    if p_phish >= 0.3:
        return "🟡 Low risk"
    return "🟢 Very low risk"

# Cached model loading
@st.cache_resource
def get_baseline_predictor():
    return BaselinePredictor()

@st.cache_resource
def get_distilbert_predictor():
    return DistilBertPredictor()

# Page setup
st.set_page_config(page_title="Veridexia", layout="wide")
st.title("Veridexia- An AI Based Phishing Detector & Explaliner")

# Session defaults
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "distilbert"
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.50
if "show_explain" not in st.session_state:
    st.session_state.show_explain = True
if "advanced_mode" not in st.session_state:
    st.session_state.advanced_mode = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Sidebar - quick controls 
with st.sidebar:
    st.header("Quick Controls")
    st.session_state.model_choice = st.selectbox(
        "Default Model",
        ["distilbert", "baseline"],
        index=0 if st.session_state.model_choice == "distilbert" else 1,
        help="DistilBERT is usually better semantically; Baseline is faster."
    )
    st.session_state.threshold = st.slider(
        "Phishing threshold",
        0.0, 1.0, float(st.session_state.threshold), 0.01,
        help="Lower = catches more phishing (more false positives). Higher = stricter."
    )
    st.session_state.show_explain = st.toggle("Show explanation", value=st.session_state.show_explain)
    st.session_state.advanced_mode = st.toggle("Advanced mode", value=st.session_state.advanced_mode)

    st.divider()
    st.caption("Files status")
    st.write("✅ dataset_stats.json" if DATASET_STATS.exists() else "⚠️ dataset_stats.json missing")
    st.write("✅ baseline external metrics" if BASELINE_METRICS.exists() else "⚠️ baseline external metrics missing")
    st.write("✅ distilbert external metrics" if DISTILBERT_METRICS.exists() else "⚠️ distilbert external metrics missing")

# Tabs
tab_dashboard, tab_scan, tab_analysis, tab_settings = st.tabs(
    ["Dashboard", "Scan Email", "Model Analysis", "Settings"]
)

# Dashboard
with tab_dashboard:
    col1, col2 = st.columns([1.2, 1])

    baseline_metrics = load_json(BASELINE_METRICS)
    distilbert_metrics = load_json(DISTILBERT_METRICS)
    stats = load_json(DATASET_STATS)

    with col1:
        st.subheader("System Overview")

        k1, k2, k3, k4 = st.columns(4)
        b_kpi = get_external_kpis(baseline_metrics)
        d_kpi = get_external_kpis(distilbert_metrics)

        if d_kpi:
            k1.metric("DistilBERT F1 (phish)", fmt_pct(d_kpi.get("f1_phish")))
            k2.metric("DistilBERT Recall (phish)", fmt_pct(d_kpi.get("recall_phish")))
        else:
            k1.metric("DistilBERT F1 (phish)", "—")
            k2.metric("DistilBERT Recall (phish)", "—")

        if b_kpi:
            k3.metric("Baseline F1 (phish)", fmt_pct(b_kpi.get("f1_phish")))
            k4.metric("Baseline Accuracy", fmt_pct(b_kpi.get("accuracy")))
        else:
            k3.metric("Baseline F1 (phish)", "—")
            k4.metric("Baseline Accuracy", "—")

        st.divider()

        st.subheader("Dataset Snapshot")
        if stats:
            ds = stats.get("dataset_info", {})
            cd = stats.get("class_distribution", {})
            fs = stats.get("feature_statistics", {})

            cA, cB, cC, cD = st.columns(4)
            cA.metric(
                "Total emails",
                f"{ds.get('total_samples', '—'):,}" if isinstance(ds.get("total_samples"), int) else "—"
            )
            cB.metric(
                "Train / Val / Test",
                f"{ds.get('train_samples','—')} / {ds.get('val_samples','—')} / {ds.get('test_samples','—')}"
            )
            cC.metric("Phishing ratio", fmt_pct(cd.get("phishing_ratio")))
            cD.metric(
                "Imbalance ratio",
                f"{cd.get('imbalance_ratio', '—'):.3f}" if isinstance(cd.get("imbalance_ratio"), (int, float)) else "—"
            )

            st.caption("Feature averages (from your preprocessing):")
            f1c, f2c, f3c = st.columns(3)
            f1c.metric(
                "Avg URLs (phishing)",
                f"{fs.get('avg_urls_phishing', '—'):.2f}" if isinstance(fs.get("avg_urls_phishing"), (int, float)) else "—"
            )
            f2c.metric(
                "Avg URLs (legit)",
                f"{fs.get('avg_urls_legitimate', '—'):.2f}" if isinstance(fs.get("avg_urls_legitimate"), (int, float)) else "—"
            )
            f3c.metric(
                "Avg urgency (phishing)",
                f"{fs.get('avg_urgency_phishing', '—'):.2f}" if isinstance(fs.get("avg_urgency_phishing"), (int, float)) else "—"
            )
        else:
            st.info("dataset_stats.json not found yet. Run your dataset preparation script to generate it.")

        if st.session_state.last_result:
            st.divider()
            st.subheader("Last Scan")
            st.write(st.session_state.last_result)

    with col2:
        st.subheader("Key Visuals")

        if BASELINE_CM_IMG.exists():
            st.image(str(BASELINE_CM_IMG), caption="Baseline confusion matrix (TREC-06)", use_container_width=True)
        else:
            st.caption("Baseline confusion matrix image not found (generate external PNGs).")

        if DISTILBERT_CM_IMG.exists():
            st.image(str(DISTILBERT_CM_IMG), caption="DistilBERT confusion matrix (TREC-06)", use_container_width=True)
        else:
            st.caption("DistilBERT confusion matrix image not found (generate external PNGs).")

# Scan Email
with tab_scan:
    st.subheader("Scan Email")

    left, right = st.columns([1.2, 1])

    with left:
        model_choice = st.selectbox(
            "Model",
            ["distilbert", "baseline"],
            index=0 if st.session_state.model_choice == "distilbert" else 1
        )
        threshold = st.slider(
            "Phishing threshold",
            0.0, 1.0, float(st.session_state.threshold), 0.01
        )
        show_explain = st.checkbox("Show explanation", value=st.session_state.show_explain)
        advanced_mode = st.checkbox("Advanced mode", value=st.session_state.advanced_mode)

        text = st.text_area(
            "Paste email content here",
            height=260,
            placeholder="Paste full email body (and subject if you have it)."
        )

        run = st.button("Analyze", type="primary", use_container_width=True)

    with right:
        st.subheader("Results")

        if run:
            if not text.strip():
                st.warning("Paste some email content first.")
                st.stop()

            start = time.time()

            #Predict (PredictionResult dataclass) 
            if model_choice == "baseline":
                predictor = get_baseline_predictor()
                result = predictor.predict(text, threshold=threshold)
                evidence = baseline_evidence(text, predictor)
            else:
                predictor = get_distilbert_predictor()
                # To keep the UI responsive
                result = predictor.predict(text, threshold=threshold, max_length=128)
                evidence = None
            
            rules = rule_based_evidence(text)
            summary = generate_explanation(result, evidence=evidence)
            st.info(summary)
            st.subheader("Detected signals")
            st.json(rules)

            ms = (time.time() - start) * 1000.0
 
            p_phish = float(result.prob_phishing)
            is_phish = (result.label == 1)

            #Headline verdict 
            st.markdown(f"### **{verdict_label(is_phish)}**")
            st.write(severity_color(p_phish, threshold))
            st.progress(min(max(p_phish, 0.0), 1.0))
            st.caption(f"Phishing probability: {p_phish:.4f} • Threshold: {threshold:.2f}")
            st.caption(f"Model: {result.model_name} • Inference time: {ms:.1f} ms")

            if advanced_mode:
                st.json({
                    "label": result.label,
                    "prob_phishing": p_phish,
                    "threshold": result.threshold,
                    "model_name": result.model_name,
                    "evidence": evidence
                })

            #Explanation
            if show_explain:
                 st.divider()
                 st.subheader("Explanation")

                 summary = generate_explanation(result, evidence=evidence)
                 st.info(summary)

                 st.subheader("Detected signals")
                 st.json(rules)

                 if evidence:
                     st.subheader("Model evidence")
                     st.json(evidence)


            if model_choice == "distilbert":
                    st.caption("Token-level evidence for DistilBERT can be added next.")

            #Save last result for dashboard 
            st.session_state.last_result = {
                "model": result.model_name,
                "threshold": float(threshold),
                "is_phishing": bool(is_phish),
                "phishing_probability": float(p_phish),
                "inference_ms": float(ms),
            }

# Model Analysis
with tab_analysis:
    st.subheader("Model Analysis & Comparison (External: TREC-06)")

    baseline_metrics = load_json(BASELINE_METRICS)
    distilbert_metrics = load_json(DISTILBERT_METRICS)

    rows = []

    if baseline_metrics:
        k = get_external_kpis(baseline_metrics)
        if k:
            rows.append({
                "Model": "Baseline (TF-IDF + Logistic Regression)",
                "Accuracy": k.get("accuracy"),
                "Precision (phish)": k.get("precision_phish"),
                "Recall (phish)": k.get("recall_phish"),
                "F1 (phish)": k.get("f1_phish"),
            })

    if distilbert_metrics:
        k = get_external_kpis(distilbert_metrics)
        if k:
            rows.append({
                "Model": "DistilBERT",
                "Accuracy": k.get("accuracy"),
                "Precision (phish)": k.get("precision_phish"),
                "Recall (phish)": k.get("recall_phish"),
                "F1 (phish)": k.get("f1_phish"),
            })

    if rows:
        df = pd.DataFrame(rows)
        for col in ["Accuracy", "Precision (phish)", "Recall (phish)", "F1 (phish)"]:
            df[col] = df[col].apply(lambda x: fmt_pct(x) if isinstance(x, (int, float)) else "—")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("External metrics not found yet. Add files to results/external/.")

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Baseline visual (external)")
        if BASELINE_CM_IMG.exists():
            st.image(str(BASELINE_CM_IMG), use_container_width=True)
        else:
            st.caption("Baseline confusion matrix image not found.")

    with c2:
        st.subheader("DistilBERT visual (external)")
        if DISTILBERT_CM_IMG.exists():
            st.image(str(DISTILBERT_CM_IMG), use_container_width=True)
        else:
            st.caption("DistilBERT confusion matrix image not found.")

    st.caption("ROC curve and feature-importance plots are hidden in external-validation view.")

# Settings
with tab_settings:
    st.subheader("Settings")
    st.write("These settings change default behaviour across the app (and help make it look deployable).")

    s1, s2 = st.columns([1, 1])

    with s1:
        st.session_state.model_choice = st.radio(
            "Default model",
            ["distilbert", "baseline"],
            index=0 if st.session_state.model_choice == "distilbert" else 1
        )
        st.session_state.threshold = st.slider(
            "Default threshold",
            0.0, 1.0, float(st.session_state.threshold), 0.01
        )
        st.session_state.show_explain = st.checkbox("Show explanations by default", value=st.session_state.show_explain)
        st.session_state.advanced_mode = st.checkbox("Enable advanced mode by default", value=st.session_state.advanced_mode)

    with s2:
        st.markdown("### Recommended presets")
        if st.button("🔒 Safer (fewer false positives)"):
            st.session_state.threshold = 0.70
        if st.button("🎣 More sensitive (catch more phishing)"):
            st.session_state.threshold = 0.40
        if st.button("🧪 Debug-friendly"):
            st.session_state.advanced_mode = True
            st.session_state.show_explain = True

        st.divider()
        st.markdown("### File locations")
        st.code(
            "\n".join([
                f"dataset_stats: {DATASET_STATS}",
                f"baseline metrics: {BASELINE_METRICS}",
                f"distilbert metrics: {DISTILBERT_METRICS}",
                f"baseline cm: {BASELINE_CM_IMG}",
                f"distilbert cm: {DISTILBERT_CM_IMG}",
            ]),
            language="text",
        )

    st.divider()
