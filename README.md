# AHEL

> A data-driven, AI-supported framework that measures and predicts the professional readiness of Computer Science students before they enter the job market.

🏆 **Achievement:** Awarded Best Senior Project by the College of Computer Science & Engineering, ranking among the **Top 3 projects university-wide**.

##  Overview

Computer Science students increasingly graduate with strong theoretical knowledge but limited practical exposure, leaving them uncertain about how ready they actually are for the job market. Traditional academic assessment measures grades, not applied, market-relevant competencies.

AHEL addresses this gap by combining student self-assessment data, live job-market analysis, and developer-survey benchmarking into a single **Readiness Index**. Using natural language processing, statistical machine learning, and an optional LLM-assisted recommendation layer, AHEL identifies each student's skill gaps relative to industry demand and their chosen specialization, then returns personalized, actionable recommendations.

##  Objectives

- Identify the gap between academic preparation and current labor-market skill demand.
- Develop a structured, quantifiable framework for measuring student readiness across practical exposure, skill alignment, experience depth, and confidence.
- Improve the accuracy of readiness assessment by benchmarking students against real job postings and a large developer survey dataset.
- Provide meaningful, specialization-specific insights and personalized development recommendations.

##  Features

- Four-step guided assessment flow: specialization selection, technical skills, practical exposure, and confidence rating.
- NLP-based skill extraction and frequency/TF-IDF analysis from real job postings.
- Statistical benchmarking against a large developer survey (skill diversity, experience, employment status).
- Specialization-aware scoring — Data Science, AI/ML, Software Engineering, Networks, and Cybersecurity each weight skills differently.
- Composite Readiness Index with a full breakdown by sub-metric.
- Dataset-backed evidence section showing how each score was derived.
- Optional AI-generated summary and recommendations via an LLM backend, with an offline rule-based fallback when no API key is configured.

##  Technologies

- Python
- Pandas
- Scikit-learn
- SciPy
- Matplotlib
- Flask & Flask-CORS
- OpenAI API (optional LLM layer)
- Joblib, python-dotenv
- HTML / CSS / JavaScript (frontend)

##  Dataset

Three datasets are used together to build the Readiness Index:

| Dataset | Description | Use |
|---|---|---|
| `Data_jobs.csv` | Real job-posting listings for computing roles | Job-market skill demand analysis |
| `survey_results_public.csv` | Large-scale developer survey (Stack Overflow Developer Survey) | Benchmarking skill diversity and experience against employment outcomes |
| `student_survey.csv` | Student practical-exposure survey | Measuring real practical exposure among peers |

**Source:** Stack Overflow Developer Survey 2025; scraped job postings; original student survey collected for this project.

Data was cleaned, encoded, normalized, and text-processed before analysis and model training.

##  Methodology

1. Data Collection (student survey, job postings, developer survey)
2. Data Cleaning
3. Data Preprocessing (encoding, normalization, text preparation)
4. Exploratory Data Analysis
5. Feature Engineering (e.g., Skill Diversity Index)
6. NLP-Based Skill Extraction (TF-IDF, skill-alias dictionary)
7. Model Training (Logistic Regression)
8. Composite Readiness Index Calculation & Results Analysis

##  Machine Learning Model

The project uses **Logistic Regression** to test whether skill diversity and experience are associated with full-time employment status (Hypothesis H2). Logistic Regression was chosen for its interpretability — its coefficients directly show the direction and strength of each feature's relationship with employment outcome, which matters for an academic readiness framework meant to explain *why* a gap exists, not just predict it.

Features used: Skill Diversity Index, Skill Count, Years Coding, Years Coding Professionally.

In addition, a lightweight **TF-IDF / NLP pipeline** extracts and weights in-demand technical skills from job postings, and an **optional LLM layer** (via the OpenAI API) generates natural-language recommendations from the computed skill gaps.

##  Results

| Metric | Score |
|---|---:|
| Accuracy | 47% |
| Precision (weighted) | 95% |
| Recall (weighted) | 47% |
| F1-Score (weighted) | 59% |
| ROC-AUC | 0.76 |

> Note: the dataset is heavily skewed toward full-time employment. H2 was **partially supported** — professional experience (years coding professionally) showed a clear positive association with full-time employment, while the Skill Diversity Index did not show the expected positive direction in this sample. Results should be interpreted with this class imbalance in mind.

##  Installation

Clone the repository:

```bash
git clone https://github.com/USERNAME/AHEL.git
cd AHEL
```

Install backend dependencies:

```bash
cd backend
pip install -r requirements.txt
```

## Usage

AHEL runs as two components that must be started in **two separate terminals**.

**Terminal 1 — Backend:**

```bash
cd backend
python server.py
```

Verify the model and LLM status in your browser:

```
http://127.0.0.1:5000/api/model-status
http://127.0.0.1:5000/api/llm-status
```

**Terminal 2 — Frontend:**

```bash
python -m http.server 8000
```

Open the app:

```
http://localhost:8000
```

##  Project Structure

```
AHEL/
│
├── analysis/            # NLP, statistical, and ML analysis scripts
├── backend/             # Flask server, LLM engine, model service
├── data/                # Job postings, developer survey, student survey
├── models/              # Trained Logistic Regression model, TF-IDF vectorizer, weights
├── outputs/             # Generated reports and analysis results
├── index.html
├── app.js
├── styles.css
├── requirements.txt
└── README.md
```

##  Future Improvements

- Replace the sample student survey with full, final collected responses for higher statistical power.
- Expand specialization coverage and refine skill-weighting per track.
- Automate job-posting collection for continuously updated market benchmarks.
- Address class imbalance in the employment-prediction model to improve accuracy.
- Deploy the framework publicly as a hosted web application.

## 👩🏻‍💻 Author

**Juhaymah Mnuif Lughaysim, CDMP®**

- LinkedIn: [linkedin.com/in/juhaymah-mnuif-cdmp®-48b176347](https://www.linkedin.com/in/juhaymah-mnuif-cdmp%C2%AE-48b176347)

**Project Team:** Renad AlRshidi, Amal Alwahbi, Ghaliah Aldhfeeri, Aljohara Nawar
**Supervisor:** Dr. Fatima Yasen — Computer Science & Engineering Department
