# AHEL - AI-Powered Data Science Readiness Analyzer

AHEL is a web-based readiness analyzer for computing and data-related careers. It combines user questionnaire input with job-market analysis, developer survey benchmarking, specialization-specific scoring, and optional LLM-generated recommendations.

## What is included

### Frontend application

- Landing page with project description
- Start Analysis button
- Four-step assessment flow
- Specialization selection
- Technical skills selection
- Practice and exposure questions
- Confidence rating
- Readiness score output
- Dataset evidence section
- Dynamic recommendations
- Optional LLM summary button

### Analysis and ML files

The project now includes a dedicated `analysis/` folder:

| File | Purpose |
|---|---|
| `analysis/job_market_analysis.py` | NLP preprocessing, skill extraction, skill frequency table, TF-IDF terms from `Data_jobs.csv` |
| `analysis/developer_survey_logistic_regression.py` | Skill Diversity Index and Logistic Regression for H2 using the developer survey |
| `analysis/student_survey_analysis.py` | Descriptive statistics and correlation analysis for H1 |
| `analysis/run_all.py` | Runs all analysis files and combines the reports |

Generated outputs are stored in `outputs/`.

## Datasets

| Dataset | Use |
|---|---|
| `data/Data_jobs.csv` | Job-market demand analysis and skill-gap comparison |
| `data/survey_results_public.csv` | Developer/professional benchmark analysis |
| `data/student_survey.csv` | Student practical exposure and readiness analysis for H1 |

> `student_survey.csv` contains the final collected student survey responses (208 responses) used to compute the Practical Exposure Benchmark and test H1 (see `outputs/student_survey_h1_report.md`).

## AI and ML models used

### 1. NLP Skill Extraction Model

This is a lightweight NLP model, not a large language model.

It performs:

- Lowercase normalization
- Symbol cleanup
- Tokenization
- Stop-word filtering
- Skill alias matching
- CountVectorizer frequency extraction
- TF-IDF term weighting

Used in:

- `analysis/job_market_analysis.py`
- Frontend dataset matching logic

### 2. Statistical ML Model

This is a traditional machine learning model.

Used in:

- `analysis/developer_survey_logistic_regression.py`

It implements:

- Skill Diversity Index
- Logistic Regression
- Accuracy
- Confusion matrix
- ROC-AUC
- Model coefficients

### 3. Optional LLM Recommendation Model

AHEL now includes optional LLM integration through:

`backend/server.py`

The frontend calls:

`POST /api/recommendations`

The backend can use an OpenAI-compatible model such as:

`gpt-4.1-mini`

This is the only large-model component. It does not run inside the browser. It runs on the backend when an API key is configured.

If no API key is configured, the backend returns a local dynamic recommendation fallback.

## Is AHEL using large models?

Partly, depending on deployment mode.

| Component | Large model? |
|---|---|
| NLP skill extraction | No |
| Job skill frequency analysis | No |
| Logistic regression | No |
| Statistical benchmarking | No |
| Browser scoring engine | No |
| Optional recommendation endpoint | Yes, if `OPENAI_API_KEY` is configured |

So the correct academic description is:

> AHEL uses lightweight NLP, statistical machine learning, and optional LLM-assisted recommendation generation.

## Scoring formula

Final Readiness Index:

```text
Readiness Index = PE × 0.25 + SDA × 0.25 + ED × 0.20 + CPR × 0.15 + SPA × 0.15
```

Where:

| Metric | Meaning | Weight |
|---|---|---|
| PE | Practical Exposure | 25% |
| SDA | Skill Demand Alignment | 25% |
| ED | Experience Depth | 20% |
| CPR | Confidence and Perceived Readiness | 15% |
| SPA | Specialization Alignment | 15% |

Skill Demand Alignment now includes specialization:

```text
SDA = JDA × 0.50 + SBA × 0.20 + SPA × 0.30
```

Where:

| Metric | Meaning |
|---|---|
| JDA | Job Demand Alignment from `Data_jobs.csv` |
| SBA | Survey Benchmark Alignment from `survey_results_public.csv` |
| SPA | Specialization Alignment based on selected career track |

## Hypotheses

### H1

Practical exposure positively affects student readiness.

Implemented in:

`analysis/student_survey_analysis.py`

Outputs:

- `outputs/student_survey_descriptive_stats.csv`
- `outputs/student_survey_correlation.csv`
- `outputs/student_survey_h1_report.md`

### H2

Higher skill diversity and experience are associated with employment status.

Implemented in:

`analysis/developer_survey_logistic_regression.py`

Outputs:

- `outputs/developer_survey_processed.csv`
- `outputs/developer_logistic_coefficients.csv`
- `outputs/developer_logistic_regression_report.md`

### H3

Job-market demand analysis can identify readiness gaps.

Implemented in:

`analysis/job_market_analysis.py`

Outputs:

- `outputs/job_skill_frequency.csv`
- `outputs/job_nlp_top_terms.csv`
- `outputs/job_tfidf_terms.csv`
- `outputs/job_market_summary.md`

Combined output:

`outputs/hypothesis_results_summary.md`

## Run frontend locally

```bash
python -m http.server 8000
```

Open:

```text
http://localhost:8000
```

## Run analysis files

Install dependencies:

```bash
pip install -r requirements.txt
```

Run individual analysis files:

```bash
python analysis/job_market_analysis.py
python analysis/developer_survey_logistic_regression.py
python analysis/student_survey_analysis.py
```

Or run all:

```bash
python analysis/run_all.py
```

## Run with LLM backend

Install dependencies:

```bash
pip install -r requirements.txt
```

Set API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

Start backend:

```bash
python backend/server.py
```

Open:

```text
http://localhost:8000
```

Then use the app and click **Generate LLM Summary** on the results page.

## Deployment notes

Static deployment works on Netlify/Vercel, but the LLM button needs a backend server. For LLM support, deploy the Flask backend separately or use a server-capable host.

Do not place API keys inside frontend JavaScript. That is how keys get stolen by strangers on the internet, because apparently we needed another way to disappoint ourselves.


---

# Added Model Files for Submission

The project now includes concrete saved ML model artifacts inside the `models/` folder:

```text
models/
├── logistic_regression_model.pkl
├── tfidf_vectorizer.pkl
├── label_encoder.pkl
├── specialization_weights.json
├── skill_clusters.json
└── model_manifest.json
```

## What Each Model File Does

### `logistic_regression_model.pkl`
A trained scikit-learn Logistic Regression pipeline. It is used to test H2 by modelling the relationship between:

- Skill Diversity Index
- Skill Count
- Years Coding
- Years Professional Coding
- Employment status

### `tfidf_vectorizer.pkl`
A trained TF-IDF vectorizer used for NLP preprocessing on job descriptions. It supports lowercase normalization, stop-word removal, unigram/bigram extraction, and technical term weighting.

### `label_encoder.pkl`
Stores the encoded target labels for the Logistic Regression model.

### `specialization_weights.json`
Stores specialization-specific skill weights so the selected specialization affects scoring. For example, Python and Machine Learning carry more weight for Data Science, while networking and Linux carry more weight for Networks.

### `skill_clusters.json`
Stores semantic groups of related skills. This supports skill-gap grouping and profile interpretation.

## Model Training Script

The model artifacts are generated by:

```bash
python analysis/train_and_export_models.py
```

This script trains and exports all model files into the `models/` directory.

## Backend Model Endpoints

The backend now includes model-loading support through:

```text
backend/model_service.py
```

Available endpoints:

```text
GET  /api/model-status
POST /api/model-predict
POST /api/recommendations
```

`/api/model-predict` uses the saved Logistic Regression model and TF-IDF vectorizer.

`/api/recommendations` uses the optional LLM integration. If `OPENAI_API_KEY` is available, it calls the OpenAI API. If not, it returns a local dynamic recommendation.

## LLM Status

AHEL includes an optional Large Language Model integration through the OpenAI API. The LLM is not stored locally as a `.pkl` file because commercial LLMs are accessed through API calls rather than shipped inside the project.

To activate the LLM:

```bash
export OPENAI_API_KEY="your-api-key"
python backend/server.py
```

The current default LLM model is:

```text
gpt-4.1-mini
```

This is a small/mini LLM, not a large locally hosted model.


## Student Survey Dataset

The project includes the collected student survey as both:

- `data/student_survey.xlsx` - original Excel dataset
- `data/student_survey.csv` - converted CSV used by the browser app and analysis scripts (208 responses)

This dataset is used to test **H1: Practical exposure is positively associated with professional readiness**.

### How AHEL uses the student survey

The app calculates a Student Survey Practical Exposure Benchmark from:

- Practical Application Opportunities
- Independence in Problem Solving
- Possession of Practical Skills

This benchmark is included in the final scoring as **20% of Practical Exposure (PE)**:

```text
PE = 80% user practical exposure score + 20% student survey exposure benchmark
```

The result page now shows:

- Student Survey Practical Exposure Benchmark
- Student Survey Readiness Benchmark

The file `analysis/student_survey_analysis.py` generates:

- `outputs/student_survey_descriptive_stats.csv`
- `outputs/student_survey_correlation.csv`
- `outputs/student_survey_h1_report.md`

The analysis found a Pearson correlation of **r = 0.451** between practical exposure and perceived professional readiness, supporting H1.

---

## Complete LLM Backend Setup

The project now includes a real LLM integration module:

```text
backend/
├── server.py
├── llm_engine.py
├── model_service.py
├── requirements.txt
└── .env.example
```

### Does AHEL use an LLM?

Yes, when `OPENAI_API_KEY` is configured. The frontend sends the user profile, score breakdown, and detected skill gaps to:

```text
POST /api/recommendations
```

The Flask backend then calls the LLM through `backend/llm_engine.py` using the OpenAI API.

If no API key is configured, AHEL clearly returns a local dynamic fallback and reports:

```text
model_status: fallback_no_api_key
```

This fallback is not an LLM. It only keeps the demo usable offline.

### How to Run with LLM

1. Open the backend folder:

```bash
cd backend
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` from the example:

```bash
cp .env.example .env
```

4. Add your OpenAI API key:

```env
OPENAI_API_KEY=sk-proj-your_key_here
AHEL_LLM_MODEL=gpt-4.1-mini
```

5. Run the Flask app:

```bash
python server.py
```

6. Open the app served by Flask:

```text
http://localhost:8000
```

Do not run the frontend separately with `python -m http.server` if you want LLM calls to work through `/api/recommendations`. Use the Flask server because it serves both the frontend and backend API.

### LLM Status Endpoint

Check whether the LLM is active:

```text
GET /api/llm-status
```

Expected response with API key:

```json
{
  "llm_available": true,
  "requires": "OPENAI_API_KEY",
  "configured_model": "gpt-4.1-mini",
  "engine_file": "backend/llm_engine.py"
}
```

### ML Model Files Included

```text
models/
├── logistic_regression_model.pkl
├── tfidf_vectorizer.pkl
├── label_encoder.pkl
├── specialization_weights.json
├── skill_clusters.json
└── model_manifest.json
```

These support logistic regression prediction, TF-IDF NLP feature extraction, specialization weighting, and semantic skill grouping.


## Specialization Scoring Fix

The Specialization field now directly affects the final score. The application uses specialization-specific priority skills and weights. For example, Data Science gives higher weight to Python, SQL, Data Analysis and Machine Learning, while Networks gives higher weight to Networking, Cybersecurity, Cloud and Docker.

Specialization affects:

- Job Demand Alignment (JDA)
- Survey Benchmark Alignment (SBA)
- Specialization Alignment (SPA)
- Skill gap ranking
- Recommendation text
- Backend `/api/model-predict` specialization coverage output

The final Skill Demand Alignment is calculated as:

```text
SDA = JDA × 0.50 + SBA × 0.20 + SPA × 0.30
```

SPA changes when a different specialization is selected, even when the same skills are selected. SPA now also contributes 15% directly to the final score, so the score visibly changes when the user changes or deselects specialization. Clicking the same specialization again deselects it.

---

## Fixed Backend Run Instructions

Run the backend first:

```bash
cd backend
pip install -r requirements.txt
python server.py
```

Backend URL:

```text
http://127.0.0.1:5000
```

Check model loading:

```text
http://127.0.0.1:5000/api/model-status
```

Expected result should show:

```json
{
  "logistic_regression_loaded": true,
  "tfidf_vectorizer_loaded": true,
  "specialization_weight_sets": ["Data Science", "Software Engineering", "Networks", "AI/ML", "Cybersecurity"],
  "skill_cluster_sets": ["Data Science", "Software Engineering", "Networks", "AI/ML", "Cybersecurity"]
}
```

Run frontend in a second terminal from the main project folder:

```bash
python -m http.server 8000
```

Open:

```text
http://localhost:8000
```

## LLM API Key

Create this file:

```text
backend/.env
```

Add:

```env
OPENAI_API_KEY=your_new_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

Do not place the API key in frontend JavaScript.

## Windows Pickle Compatibility Fix

If `/api/model-status` shows `ModuleNotFoundError: No module named 'numpy._core'`, the model files were created with a different NumPy version. This project now fixes that automatically: when the backend starts, it tries to load the `.pkl` files; if loading fails, it regenerates compatible model files from the local datasets using your current Python environment.

You can also trigger regeneration manually:

```text
http://127.0.0.1:5000/api/regenerate-models
```

After regeneration, check:

```text
http://127.0.0.1:5000/api/model-status
```

Expected result:

```json
{
  "logistic_regression_loaded": true,
  "label_encoder_loaded": true,
  "tfidf_vectorizer_loaded": true
}
```

## Specialization Scoring Fix Verification

The specialization field is not decorative. It is used in two places:

1. `SPA` = Specialization Alignment Score
2. `SDA` = Skill Demand Alignment, where SPA contributes 30% of SDA
3. Final score, where SPA also contributes 15% directly

Current formula:

```text
SDA = JDA × 0.50 + SBA × 0.20 + SPA × 0.30
Final Score = PE × 0.25 + SDA × 0.25 + ED × 0.20 + CPR × 0.15 + SPA × 0.15
```

No specialization selected uses a neutral SPA baseline of 30. Selecting a specialization calculates SPA from the selected skills against that specialization's required priority skills. Therefore selecting, changing, or deselecting specialization changes the final score.

To verify manually:

```bash
python analysis/test_specialization_scoring.py
```

Expected behavior with Python + SQL + Machine Learning selected:

```text
Data Science / AI & ML should score higher.
Networks / Software Engineering should score lower.
Not selected should use neutral baseline.
```

## LLM API Key Validation

The backend now validates the OpenAI API key before generating live LLM recommendations.

Useful endpoints:

```text
http://127.0.0.1:5000/api/llm-status
http://127.0.0.1:5000/api/validate-openai-key
http://127.0.0.1:5000/api/model-status
```

If the key is missing, invalid, expired, or billing is unavailable, the frontend shows a clear red LLM error message and falls back to local rule-based recommendations.

Keep the key only in:

```text
backend/.env
```

Example:

```env
OPENAI_API_KEY=your_new_key_here
OPENAI_MODEL=gpt-4o-mini
```
