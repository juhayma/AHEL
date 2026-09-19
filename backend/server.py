"""AHEL Flask backend.

Runs on http://127.0.0.1:5000
Provides:
- /api/model-status
- /api/model-predict
- /api/recommendations
- /api/llm-status
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
load_dotenv(BACKEND_DIR / ".env")

try:
    from llm_engine import generate_llm_recommendation, local_dynamic_recommendation, llm_available, validate_openai_key
except Exception:
    from backend.llm_engine import generate_llm_recommendation, local_dynamic_recommendation, llm_available, validate_openai_key

try:
    from model_service import (
        model_status as get_model_status,
        predict_employment_alignment,
        extract_job_terms,
        specialization_weights,
        skill_clusters,
        specialization_coverage,
        regenerate_models_now,
    )
except Exception as exc:
    IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

    def get_model_status():
        return {
            "models_dir": str(ROOT_DIR / "models"),
            "logistic_regression_loaded": False,
            "label_encoder_loaded": False,
            "tfidf_vectorizer_loaded": False,
            "specialization_weight_sets": [],
            "skill_cluster_sets": [],
            "errors": {"model_service_import": IMPORT_ERROR},
        }

    predict_employment_alignment = None
    extract_job_terms = None
    specialization_weights = {}
    skill_clusters = {}
    specialization_coverage = None
    regenerate_models_now = None

app = Flask(__name__, static_folder=str(ROOT_DIR), static_url_path="")
CORS(app)


@app.route("/")
def index():
    return send_from_directory(ROOT_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "backend_port": 5000,
        "root_dir": str(ROOT_DIR),
        "openai_key_configured": bool(os.getenv("OPENAI_API_KEY")),
    })


@app.route("/api/model-status", methods=["GET"])
def api_model_status():
    status = get_model_status()
    llm_key_status = validate_openai_key(live_check=False)
    status["llm_requires"] = "OPENAI_API_KEY environment variable"
    status["openai_key_configured"] = bool(os.getenv("OPENAI_API_KEY"))
    status["openai_key_valid"] = llm_key_status.get("valid")
    status["openai_key_status"] = llm_key_status.get("status")
    status["openai_key_message"] = llm_key_status.get("message")
    status["configured_model"] = llm_key_status.get("model")
    return jsonify(status)


@app.route("/api/regenerate-models", methods=["POST", "GET"])
def api_regenerate_models():
    if regenerate_models_now is None:
        return jsonify({
            "error": "Model regeneration function is not available.",
            "model_status": get_model_status(),
        }), 500
    return jsonify(regenerate_models_now())


@app.route("/api/model-predict", methods=["POST"])
def model_predict():
    if predict_employment_alignment is None:
        return jsonify({
            "error": "Model service is not available.",
            "model_status": get_model_status(),
        }), 500

    data = request.get_json(force=True) or {}
    profile = data.get("profile", {}) or {}
    skills = profile.get("skills", []) or []
    skill_count = int(profile.get("skill_count", len(skills)))
    skill_diversity_index = float(profile.get("skill_diversity_index", min(100, skill_count * 10)))
    years_coding = float(profile.get("years_coding", 1))
    years_professional = float(profile.get("years_professional", 0))
    job_text = data.get("job_text", " ".join(skills))
    specialization = profile.get("specialization", data.get("specialization", "Data Science"))

    prediction = predict_employment_alignment(
        skill_diversity_index=skill_diversity_index,
        skill_count=skill_count,
        years_coding=years_coding,
        years_professional=years_professional,
    )

    return jsonify({
        "employment_alignment_prediction": prediction,
        "top_tfidf_terms": extract_job_terms(job_text) if extract_job_terms else [],
        "specialization_coverage": specialization_coverage(specialization, skills) if specialization_coverage else None,
        "specialization_weights_available": list(specialization_weights.keys()),
        "skill_clusters_available": list(skill_clusters.keys()),
    })


@app.route("/api/recommendations", methods=["POST"])
def recommendations():
    data = request.get_json(force=True) or {}
    profile = data.get("profile", {}) or {}
    score = data.get("score", {}) or {}
    gaps = data.get("gaps", []) or []

    key_status = validate_openai_key(live_check=True)
    if not key_status.get("valid"):
        fallback = local_dynamic_recommendation(profile, score, gaps)
        fallback["model_status"] = "llm_unavailable"
        fallback["provider"] = "Local rule-based fallback"
        fallback["llm_error"] = key_status.get("message")
        fallback["openai_key_valid"] = False
        fallback["openai_key_status"] = key_status.get("status")
        return jsonify(fallback), 401

    try:
        result = generate_llm_recommendation(profile, score, gaps)
        result["openai_key_valid"] = True
        return jsonify(result)
    except Exception as exc:
        fallback = local_dynamic_recommendation(profile, score, gaps)
        fallback["model_status"] = "llm_failed_after_validation"
        fallback["llm_error"] = f"{type(exc).__name__}: {exc}"
        fallback["openai_key_valid"] = False
        return jsonify(fallback), 502


@app.route("/api/llm-status", methods=["GET"])
def llm_status():
    key_status = validate_openai_key(live_check=True)
    return jsonify({
        "llm_available": bool(key_status.get("valid")),
        "requires": "OPENAI_API_KEY",
        "openai_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        "openai_key_valid": key_status.get("valid"),
        "openai_key_status": key_status.get("status"),
        "openai_key_message": key_status.get("message"),
        "configured_model": key_status.get("model"),
        "engine_file": "backend/llm_engine.py",
    })


@app.route("/api/validate-openai-key", methods=["GET"])
def validate_key_endpoint():
    return jsonify(validate_openai_key(live_check=True))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
