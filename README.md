# Veridexia: ML-Based Phishing Detection
Veridexia is a final year Computer Science (Cyber Security) project for COMP 3000. This is a Streamlit-based prototype for detecting phishing emails. It sorts emails into two groups: phishing and legitimate and it gives users evidence-based explanations to help them understand why the email was classified that way.

The project looks at the difference between a classical TF-IDF + Logistic Regression baseline and a DistilBERT transformer model. There is also a rule-supported explanation layer, email and document analysis based on uploads, an optional read-only Gmail import and a model analysis view for internal and external validation results.

## Project Vision
Phishing remains a significant cybersecurity threat because it exploits human judgement as well as technical weaknesses. The aim of this project is to build a usable prototype that does more than output a label: it should support decision-making by showing why an email may be suspicious.

The project focuses on three goals:

- compare a lightweight classical baseline with a DistilBERT model
- provide clear, easy to understand explanations alongside predictions.
- evaluate model behaviour using both in-domain results and external validation.

## Supervisor

Shaymaa Al-Juboori

## How To Run The Project

Clone the repository and open a terminal in the project root directory.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the Streamlit app:

```powershell
python -m streamlit run app\app.py
```

If you do not want to activate the virtual environment manually, run Streamlit through the venv Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app\app.py
```

Using python -m streamlit is recommended because it ensures Streamlit runs with the same Python environment where the dependencies are installed.

## Dependencies And Installation
The main dependencies are listed in requirements.txt. They include:

streamlit for the web prototype
pandas, numpy and scikit-learn for data processing and the baseline model
transformers, datasets, torch and accelerate for DistilBERT training/inference
beautifulsoup4, pypdf, Pillow and pytesseract for upload text extraction
Google API packages for optional read-only Gmail import
Gmail import requires local OAuth files. The files credentials.json and token.json are intentionally ignored by Git and should not be committed.

## Repository Structure
```text
app/        Streamlit user interface
src/        inference, explanation, Gmail import, and upload extraction logic
models/     saved baseline and DistilBERT model artefacts
results/    evaluation outputs and figures
data/       generated dataset statistics and processed artefacts
notebooks/  exploratory analysis notebook
scripts/    training, evaluation and utility scripts
```

## Models And Data
The primary internal dataset is the Kaggle Phishing Email Dataset. It is used for model development, preprocessing, training, validation and in-domain testing.

Two model approaches are compared:

Baseline: TF-IDF vectorisation with Logistic Regression and balanced class weights.
DistilBERT: distilbert-base-uncased fine-tuned for binary phishing/legitimate classification.
External validation is performed using TREC-06 to test generalisation on a separate dataset. Although TREC-06 is older, it provides a useful cross-dataset check because strong in-domain scores can be overly optimistic.

The application also uses a rule-supported evidence layer to identify visible signals such as urgency language, credential requests, suspicious links, sender/domain mismatches, and contextual reassurance signals. This layer supports explanation and review, but the ML models remain the main classifiers.

## Development And Commit Approach
The project was developed iteratively using sprints. Commits were generally made after completing a sprint or meaningful project milestones, rather than after every small local edit. As a result, the commit history is less clustered with micro commits.

## Security And Privacy Notes
This is a local application rather than a production email security product. It does not continuously monitor inboxes, quarantine emails, or perform live enterprise deployment.
