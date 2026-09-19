"""Job Market Dataset Analysis for AHEL.

Outputs:
- outputs/job_skill_frequency.csv
- outputs/job_market_summary.md

Implements NLP preprocessing: normalization, tokenization, stop-word filtering,
skill alias extraction, and frequency table generation from Data_jobs.csv.
"""
from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Data_jobs.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SKILL_ALIASES = {
    "Python": ["python"],
    "SQL": ["sql", "structured query language"],
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Excel": ["excel", "spreadsheet"],
    "Machine Learning": ["machine learning", "ml", "scikit", "regression", "classification"],
    "Deep Learning": ["deep learning", "neural", "tensorflow", "pytorch", "keras"],
    "NLP": ["nlp", "natural language processing", "text mining", "topic tagging", "language model"],
    "Data Analysis": ["data analysis", "analytics", "statistics", "statistical", "visualization"],
    "Cloud": ["aws", "azure", "gcp", "google cloud", "cloud"],
    "Docker/Kubernetes": ["docker", "kubernetes", "container"],
    "Big Data": ["spark", "hadoop", "databricks", "big data"],
    "MLOps": ["mlops", "model deployment", "airflow", "kubeflow", "ci cd"],
    "Git": ["git", "github", "version control"],
    "Cybersecurity": ["security", "cybersecurity", "siem", "penetration"],
    "Networking": ["network", "networking", "routing", "switching", "tcp", "cisco"],
}

STOP_WORDS = set("""
and or the a an to of in with for on by is are as this that from at be you your we our will can should using use used experience required role team data work business skills more less show
""".split())


def normalize(text: str) -> str:
    text = str(text or "").lower().replace("c++", "cpp")
    text = re.sub(r"[^a-z0-9+#.]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    return [t for t in normalize(text).split() if t not in STOP_WORDS and len(t) > 1]


def contains_alias(text: str, alias: str) -> bool:
    clean = normalize(text)
    alias_clean = normalize(alias)
    if not alias_clean:
        return False
    if " " in alias_clean:
        return alias_clean in clean
    return f" {alias_clean} " in f" {clean} "


def main():
    jobs = pd.read_csv(DATA)
    jobs["combined_text"] = jobs[[c for c in ["title", "description", "work_type", "employment_type"] if c in jobs.columns]].fillna("").agg(" ".join, axis=1)
    jobs["tokens"] = jobs["combined_text"].apply(tokenize)
    jobs["processed_text"] = jobs["tokens"].apply(lambda x: " ".join(x))

    rows = []
    for skill, aliases in SKILL_ALIASES.items():
        count = int(jobs["combined_text"].apply(lambda txt: any(contains_alias(txt, a) for a in aliases)).sum())
        rows.append({"skill": skill, "job_count": count, "percentage_of_jobs": round((count / len(jobs)) * 100, 2)})

    freq = pd.DataFrame(rows).sort_values("job_count", ascending=False)
    freq.to_csv(OUT / "job_skill_frequency.csv", index=False)

    vectorizer = CountVectorizer(max_features=40, ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(jobs["processed_text"])
    term_freq = pd.DataFrame({"term": vectorizer.get_feature_names_out(), "frequency": matrix.sum(axis=0).A1}).sort_values("frequency", ascending=False)
    term_freq.to_csv(OUT / "job_nlp_top_terms.csv", index=False)

    tfidf = TfidfVectorizer(max_features=40, ngram_range=(1, 2), stop_words="english")
    tfidf_matrix = tfidf.fit_transform(jobs["processed_text"])
    tfidf_terms = pd.DataFrame({"term": tfidf.get_feature_names_out(), "tfidf_weight": tfidf_matrix.mean(axis=0).A1}).sort_values("tfidf_weight", ascending=False)
    tfidf_terms.to_csv(OUT / "job_tfidf_terms.csv", index=False)

    top = freq.head(10).to_markdown(index=False)
    summary = f"""# Job Market Dataset Analysis

Dataset: `Data_jobs.csv`  
Records analyzed: **{len(jobs)}**

## NLP Preprocessing Implemented

- Lowercase normalization
- Symbol cleanup
- Tokenization
- Stop-word filtering
- Skill alias matching
- CountVectorizer frequency extraction
- TF-IDF term weighting

## Top Extracted Skill Frequencies

{top}

## Output Files

- `outputs/job_skill_frequency.csv`
- `outputs/job_nlp_top_terms.csv`
- `outputs/job_tfidf_terms.csv`

## Hypothesis H3 Evidence

H3 states that job-market skill demand can be used to identify readiness gaps. The extracted frequency table provides the job-market evidence used by the AHEL frontend to compare user-selected skills with market demand.
"""
    (OUT / "job_market_summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
