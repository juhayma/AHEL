"""Robust model loading and inference utilities for AHEL.

This version fixes Windows pickle compatibility issues such as:
ModuleNotFoundError: No module named 'numpy._core'

If old .pkl files cannot be loaded, the backend automatically regenerates
fresh model files from the local datasets using the user's current Python,
NumPy and scikit-learn environment.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys
import subprocess
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
ANALYSIS_DIR = BASE_DIR / "analysis"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_ERRORS: dict[str, str] = {}
MODEL_REGENERATION: dict[str, str | bool] = {
    "attempted": False,
    "success": False,
    "message": "Not attempted"
}

REQUIRED_MODEL_FILES = [
    "logistic_regression_model.pkl",
    "label_encoder.pkl",
    "tfidf_vectorizer.pkl",
    "specialization_weights.json",
    "skill_clusters.json",
]


def _format_error(exc: Exception, path: Path) -> str:
    return f"{type(exc).__name__}: {exc}. Path checked: {path}"


def _run_training_script() -> bool:
    """Regenerate model artifacts with current environment."""
    global MODEL_REGENERATION
    MODEL_REGENERATION["attempted"] = True

    train_script = ANALYSIS_DIR / "train_and_export_models.py"
    if not train_script.exists():
        MODEL_REGENERATION["success"] = False
        MODEL_REGENERATION["message"] = f"Training script not found: {train_script}"
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(train_script)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode == 0:
            MODEL_REGENERATION["success"] = True
            MODEL_REGENERATION["message"] = "Models regenerated successfully from local datasets."
            # Clear old pkl compatibility errors after successful regeneration.
            for key in ["logistic_regression_model.pkl", "label_encoder.pkl", "tfidf_vectorizer.pkl"]:
                MODEL_ERRORS.pop(key, None)
            return True

        MODEL_REGENERATION["success"] = False
        MODEL_REGENERATION["message"] = (
            "Model regeneration failed.\n"
            f"STDOUT:\n{result.stdout[-3000:]}\n"
            f"STDERR:\n{result.stderr[-3000:]}"
        )
        return False
    except Exception as exc:
        MODEL_REGENERATION["success"] = False
        MODEL_REGENERATION["message"] = f"{type(exc).__name__}: {exc}"
        return False


def _load_joblib(filename: str):
    path = MODELS_DIR / filename
    try:
        return joblib.load(path)
    except Exception as exc:
        MODEL_ERRORS[filename] = _format_error(exc, path)
        return None


def _load_json(filename: str, default):
    path = MODELS_DIR / filename
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        MODEL_ERRORS[filename] = _format_error(exc, path)
        return default


def _load_all_models():
    logistic = _load_joblib("logistic_regression_model.pkl")
    encoder = _load_joblib("label_encoder.pkl")
    vectorizer = _load_joblib("tfidf_vectorizer.pkl")
    specs = _load_json("specialization_weights.json", {})
    clusters = _load_json("skill_clusters.json", {})
    return logistic, encoder, vectorizer, specs, clusters


logistic_model, label_encoder, tfidf_vectorizer, specialization_weights, skill_clusters = _load_all_models()

# If pickle loading failed, regenerate compatible .pkl files and reload.
if logistic_model is None or label_encoder is None or tfidf_vectorizer is None:
    regenerated = _run_training_script()
    if regenerated:
        logistic_model, label_encoder, tfidf_vectorizer, specialization_weights, skill_clusters = _load_all_models()


def regenerate_models_now() -> dict:
    """Public function used by /api/regenerate-models."""
    global logistic_model, label_encoder, tfidf_vectorizer, specialization_weights, skill_clusters
    MODEL_ERRORS.clear()
    ok = _run_training_script()
    logistic_model, label_encoder, tfidf_vectorizer, specialization_weights, skill_clusters = _load_all_models()
    return model_status()


def model_status() -> dict:
    return {
        "models_dir": str(MODELS_DIR),
        "logistic_regression_loaded": logistic_model is not None,
        "label_encoder_loaded": label_encoder is not None,
        "tfidf_vectorizer_loaded": tfidf_vectorizer is not None,
        "specialization_weight_sets": list(specialization_weights.keys()),
        "skill_cluster_sets": list(skill_clusters.keys()),
        "model_regeneration": MODEL_REGENERATION,
        "errors": MODEL_ERRORS,
    }


def predict_employment_alignment(
    skill_diversity_index: float,
    skill_count: int,
    years_coding: float,
    years_professional: float,
) -> dict:
    if logistic_model is None or label_encoder is None:
        return {
            "error": "Logistic regression model or label encoder is not loaded.",
            "model_status": model_status(),
        }

    X = [[skill_diversity_index, skill_count, years_coding, years_professional]]
    pred = logistic_model.predict(X)[0]
    proba = logistic_model.predict_proba(X)[0]
    labels = label_encoder.inverse_transform(range(len(proba)))

    return {
        "predicted_label": label_encoder.inverse_transform([pred])[0],
        "probabilities": {str(label): float(prob) for label, prob in zip(labels, proba)},
    }


def extract_job_terms(text: str, top_n: int = 15) -> list[dict]:
    if tfidf_vectorizer is None:
        return []

    matrix = tfidf_vectorizer.transform([text or ""])
    scores = matrix.toarray()[0]
    terms = tfidf_vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)

    return [
        {"term": term, "score": float(score)}
        for term, score in ranked[:top_n]
        if score > 0
    ]


def normalize_specialization_name(specialization: str | None) -> str:
    mapping = {
        "AI & ML": "AI/ML",
        "AI and ML": "AI/ML",
        "Artificial Intelligence": "AI/ML",
    }
    if not specialization:
        return ""
    return mapping.get(str(specialization), str(specialization))


def specialization_skill_weight(specialization: str, skill: str) -> float:
    spec = normalize_specialization_name(specialization)
    return float(specialization_weights.get(spec, {}).get(str(skill).lower(), 1.0))


def specialization_coverage(specialization: str, skills: list[str]) -> dict:
    spec = normalize_specialization_name(specialization)
    weights = specialization_weights.get(spec, {})
    selected = {str(s).lower() for s in skills}
    possible = sum(float(v) for v in weights.values()) or 1.0
    achieved = sum(float(w) for skill, w in weights.items() if skill in selected)
    missing = [skill for skill in weights.keys() if skill not in selected]
    matched = [skill for skill in weights.keys() if skill in selected]

    return {
        "specialization": specialization or "Not selected",
        "normalized_specialization": spec or "Not selected",
        "score": round((achieved / possible) * 100, 2),
        "matched_priority_skills": matched,
        "missing_priority_skills": missing,
        "weights_used": weights,
    }
