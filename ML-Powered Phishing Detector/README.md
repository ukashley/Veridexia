# Veridexia

Veridexia is a Streamlit-based phishing email detection prototype created for the COMP3000 final year project. It analyses email text, predicts whether a message is likely to be phishing or legitimate, and provides evidence-supported explanations to help users understand the result.

The project compares a TF-IDF + Logistic Regression baseline with a fine-tuned DistilBERT classifier. DistilBERT is used as the main model in the app, while the baseline remains available in advanced mode for comparison and fallback testing. The app supports pasted email text and uploaded saved email artefacts such as `.eml`, `.txt`, `.docx`, and `.pdf` files.

## Project Vision

The aim of Veridexia is to provide a practical phishing detection tool that does more than output a label. The system is designed to support user judgement by combining machine learning predictions with explanation cues such as urgency, credential requests, threat language, suspicious links, sender-domain issues, and contextual signals.

## How To Run The App

From the project root, create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the Streamlit application:

```powershell
python -m streamlit run app/app.py
```

Using `python -m streamlit` is recommended because it ensures Streamlit runs inside the active virtual environment.

## Gmail Import

Gmail import is intended for local demonstration. To use it, place a valid Google OAuth `credentials.json` file in the project root. The first time Gmail import is used, Google will open a browser sign-in flow and create a local `token.json` file.

Do not commit or share:

```text
credentials.json
token.json
client_secret_*.json
.streamlit/secrets.toml
```

## Repository Structure

```text
app/        Streamlit user interface
src/        inference, explanation, Gmail import, and upload extraction logic
models/     saved baseline and DistilBERT model artefacts
results/    evaluation outputs and figures
data/       generated dataset statistics and processed artefacts
notebooks/  exploratory analysis notebook
scripts/    training, evaluation, and utility scripts
```

## Models And Data

The internal training and evaluation work used a Kaggle phishing email dataset. External validation was also carried out using TREC-06 to test cross-dataset generalisation. The app uses saved local model artefacts from the `models/` directory rather than retraining models at runtime.

## Report And Demonstration

This repository supports the submitted COMP3000 report and viva demonstration. The app is designed to be run locally for marking, demonstration, and user testing.
