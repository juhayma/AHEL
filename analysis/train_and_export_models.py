"""Train and export AHEL model artifacts.

This file creates the concrete model files required for submission:
- models/logistic_regression_model.pkl
- models/tfidf_vectorizer.pkl
- models/label_encoder.pkl
- models/specialization_weights.json
- models/skill_clusters.json
- models/model_manifest.json

The Logistic Regression model tests H2 using developer survey features:
Skill Diversity Index, Skill Count, Years Coding, and Professional Coding Years.
The TF-IDF vectorizer supports NLP preprocessing over job descriptions.
"""
from __future__ import annotations

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"
MODELS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

SKILL_COLS = [
    "LanguageWorkedWith", "DatabaseWorkedWith", "PlatformWorkedWith",
    "FrameworkWorkedWith", "Methodology", "VersionControl"
]

SKILL_CLUSTERS = {
    "Data Science": ["python", "r", "sql", "machine learning", "statistics", "pandas", "numpy", "scikit-learn", "tableau", "power bi", "nlp"],
    "Software Engineering": ["javascript", "typescript", "java", "c#", "react", "node", "git", "docker", "api", "testing"],
    "Networks": ["network", "linux", "tcp", "ip", "security", "cloud", "aws", "azure", "firewall", "cisco"],
    "Cybersecurity": ["security", "linux", "network", "incident", "risk", "cloud", "python", "siem", "vulnerability"],
    "AI/ML": ["python", "machine learning", "deep learning", "nlp", "tensorflow", "pytorch", "scikit-learn", "statistics"],
    "Business Analytics": ["excel", "sql", "power bi", "tableau", "dashboard", "reporting", "statistics", "data visualization"]
}

SPECIALIZATION_WEIGHTS = {
    "Data Science": {"python": 1.6, "sql": 1.4, "machine learning": 1.7, "statistics": 1.5, "power bi": 1.2, "tableau": 1.2, "excel": 1.1},
    "Software Engineering": {"javascript": 1.6, "react": 1.5, "node": 1.4, "java": 1.3, "git": 1.3, "docker": 1.2, "sql": 1.1},
    "Networks": {"network": 1.8, "linux": 1.5, "security": 1.4, "cloud": 1.3, "aws": 1.2, "azure": 1.2, "python": 1.1},
    "Cybersecurity": {"security": 1.8, "network": 1.5, "linux": 1.4, "risk": 1.3, "python": 1.2, "cloud": 1.2},
    "AI/ML": {"python": 1.6, "machine learning": 1.8, "deep learning": 1.6, "nlp": 1.5, "statistics": 1.4, "sql": 1.1},
    "Business Analytics": {"excel": 1.5, "sql": 1.5, "power bi": 1.6, "tableau": 1.4, "statistics": 1.3, "python": 1.1}
}


def split_multi(value) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [x.strip().lower() for x in str(value).split(";") if x.strip()]


def parse_years(value) -> float:
    text = str(value).lower()
    if "30 or more" in text:
        return 30.0
    if "less than" in text:
        return 0.5
    nums = [int(x) for x in re.findall(r"\d+", text)]
    if not nums:
        return np.nan
    return float(np.mean(nums[:2])) if len(nums) >= 2 else float(nums[0])


def train_tfidf_vectorizer() -> dict:
    jobs = pd.read_csv(DATA / "Data_jobs.csv")
    text = (
        jobs.get("title", pd.Series([""] * len(jobs))).fillna("").astype(str) + " " +
        jobs.get("description", pd.Series([""] * len(jobs))).fillna("").astype(str)
    )
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=1000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\+#\.]{1,}\b"
    )
    matrix = vectorizer.fit_transform(text)
    joblib.dump(vectorizer, MODELS / "tfidf_vectorizer.pkl")
    terms = pd.DataFrame({
        "term": vectorizer.get_feature_names_out(),
        "tfidf_total": np.asarray(matrix.sum(axis=0)).ravel()
    }).sort_values("tfidf_total", ascending=False)
    terms.head(150).to_csv(OUT / "model_tfidf_top_terms.csv", index=False)
    return {"rows": int(len(jobs)), "features": int(matrix.shape[1])}


def train_logistic_regression() -> dict:
    survey_path = DATA / "survey_results_public.csv"
    if not survey_path.exists():
        survey_path = DATA / "survey_results_public.csv"
    df = pd.read_csv(survey_path)
    for col in SKILL_COLS:
        if col not in df.columns:
            df[col] = ""

    df["Skill_Count"] = df[SKILL_COLS].apply(
        lambda row: len(set(sum([split_multi(row[c]) for c in SKILL_COLS], []))), axis=1
    )
    max_skill_count = max(1, int(df["Skill_Count"].max()))
    df["Skill_Diversity_Index"] = (df["Skill_Count"] / max_skill_count * 100).round(2)
    df["YearsCoding_Num"] = df.get("YearsCoding", pd.Series([np.nan] * len(df))).apply(parse_years)
    df["YearsCodingProf_Num"] = df.get("YearsCodingProf", pd.Series([np.nan] * len(df))).apply(parse_years)

    employment = df.get("Employment", pd.Series([""] * len(df))).astype(str).str.lower()
    # Binary target used for H2: full-time employment vs other statuses.
    df["Employment_Target_Label"] = np.where(employment.str.contains("employed full-time", regex=False), "full_time", "other")

    model_df = df[["Skill_Diversity_Index", "Skill_Count", "YearsCoding_Num", "YearsCodingProf_Num", "Employment_Target_Label"]].dropna()
    le = LabelEncoder()
    y = le.fit_transform(model_df["Employment_Target_Label"])
    X = model_df[["Skill_Diversity_Index", "Skill_Count", "YearsCoding_Num", "YearsCodingProf_Num"]]

    if len(set(y)) < 2 or len(model_df) < 20:
        raise ValueError("Not enough rows/classes to train logistic regression model.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, target_names=le.classes_, zero_division=0)

    joblib.dump(pipe, MODELS / "logistic_regression_model.pkl")
    joblib.dump(le, MODELS / "label_encoder.pkl")

    df.to_csv(OUT / "developer_survey_processed.csv", index=False)
    pd.DataFrame({
        "feature": X.columns,
        "coefficient": pipe.named_steps["logistic_regression"].coef_[0]
    }).to_csv(OUT / "model_logistic_coefficients.csv", index=False)

    (OUT / "model_training_report.md").write_text(f"""# AHEL Model Training Report

## Exported Model Files

- `models/logistic_regression_model.pkl`
- `models/tfidf_vectorizer.pkl`
- `models/label_encoder.pkl`
- `models/specialization_weights.json`
- `models/skill_clusters.json`

## Logistic Regression Model

Purpose: Tests H2 by modelling the relationship between skill diversity, experience depth, and employment status.

Rows used: **{len(model_df)}**
Features:
- Skill_Diversity_Index
- Skill_Count
- YearsCoding_Num
- YearsCodingProf_Num

Accuracy: **{acc:.3f}**
ROC-AUC: **{auc:.3f}**

Confusion matrix:

```text
{cm}
```

Classification report:

```text
{report}
```

## TF-IDF Vectorizer

Purpose: NLP preprocessing and feature extraction from job titles/descriptions.

The vectorizer uses lowercase normalization, stop-word removal, unigrams/bigrams, and a restricted technical-token pattern.
""", encoding="utf-8")

    return {"rows": int(len(model_df)), "accuracy": float(acc), "roc_auc": float(auc), "classes": list(le.classes_)}


def export_configs() -> None:
    (MODELS / "specialization_weights.json").write_text(json.dumps(SPECIALIZATION_WEIGHTS, indent=2), encoding="utf-8")
    (MODELS / "skill_clusters.json").write_text(json.dumps(SKILL_CLUSTERS, indent=2), encoding="utf-8")


def main():
    tfidf_info = train_tfidf_vectorizer()
    lr_info = train_logistic_regression()
    export_configs()
    manifest = {
        "project": "AHEL Data Science Readiness Analyzer",
        "model_files": {
            "logistic_regression_model": "models/logistic_regression_model.pkl",
            "tfidf_vectorizer": "models/tfidf_vectorizer.pkl",
            "label_encoder": "models/label_encoder.pkl",
            "specialization_weights": "models/specialization_weights.json",
            "skill_clusters": "models/skill_clusters.json"
        },
        "tfidf": tfidf_info,
        "logistic_regression": lr_info,
        "llm_status": "Optional OpenAI API backend in backend/server.py. Requires OPENAI_API_KEY."
    }
    (MODELS / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
