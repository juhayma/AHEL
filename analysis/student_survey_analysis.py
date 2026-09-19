"""
Student Survey Analysis for AHEL

This script analyzes the uploaded student_survey dataset to test H1:
H1: Practical exposure is positively associated with professional readiness.

Input:
    data/student_survey.csv
    data/student_survey.xlsx is included as the original source file.

Output:
    outputs/student_survey_descriptive_stats.csv
    outputs/student_survey_correlation.csv
    outputs/student_survey_h1_report.md
"""

import csv
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "student_survey.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

MAPS = {
    "Professional Readiness": {"Not Ready": 0, "Need Improvement": 50, "Ready": 100},
    "Technical Confidence Level": {"Weak": 25, "Medium": 60, "High": 90},
    "Practical Application Opportunities": {"No": 0, "Somewhat": 50, "Yes": 100},
    "Reliance on Theory": {"Yes": 100, "Somewhat": 50, "No": 0},
    "Independence in Problem Solving": {"Rarely": 25, "Sometimes": 60, "Always": 90},
    "Possession of Practical Skills": {"No": 0, "Somewhat": 50, "Yes": 100},
    "Gap with Job Market": {"Large Gap": 0, "Moderate Gap": 50, "No Gap": 100},
}

def score(row, col):
    value = (row.get(col) or "").strip()
    if not value:
        return None
    if col in MAPS:
        return MAPS[col].get(value)
    try:
        return float(value)
    except ValueError:
        return None

def corr(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else None

with DATA.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    exposure_values = [
        score(row, "Practical Application Opportunities"),
        score(row, "Independence in Problem Solving"),
        score(row, "Possession of Practical Skills"),
    ]
    exposure_values = [x for x in exposure_values if x is not None]
    row["Practical_Exposure_Score"] = sum(exposure_values) / len(exposure_values) if exposure_values else None
    row["Professional_Readiness_Score"] = score(row, "Professional Readiness")

variables = [
    "Practical_Exposure_Score",
    "Professional_Readiness_Score",
    "Technical Confidence Level",
    "Reliance on Theory",
    "Gap with Job Market",
]

with (OUT / "student_survey_descriptive_stats.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["variable", "count", "mean", "min", "max"])
    for col in variables:
        vals = []
        for row in rows:
            v = row.get(col) if col in row else score(row, col)
            if isinstance(v, str):
                v = score(row, col)
            if v is not None:
                vals.append(float(v))
        if vals:
            writer.writerow([col, len(vals), round(statistics.mean(vals), 2), round(min(vals), 2), round(max(vals), 2)])

readiness = [row["Professional_Readiness_Score"] for row in rows]
correlations = []
for col in ["Practical_Exposure_Score", "Technical Confidence Level", "Reliance on Theory", "Gap with Job Market"]:
    xs = []
    for row in rows:
        v = row.get(col) if col in row else score(row, col)
        if isinstance(v, str):
            v = score(row, col)
        xs.append(v)
    correlations.append((col, "Professional_Readiness_Score", corr(xs, readiness)))

with (OUT / "student_survey_correlation.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["variable_x", "variable_y", "pearson_r"])
    for x, y, r in correlations:
        writer.writerow([x, y, round(r, 3) if r is not None else ""])

h1_r = next(r for x, _, r in correlations if x == "Practical_Exposure_Score")
report = f"""# Student Survey Analysis Report (H1)

Source files:
- `data/student_survey.xlsx` original uploaded dataset
- `data/student_survey.csv` browser-friendly converted dataset

Rows analyzed: {len(rows)}

## Hypothesis H1
Practical exposure is positively associated with professional readiness.

## Derived Variable
`Practical_Exposure_Score` is calculated from:
- Practical Application Opportunities
- Independence in Problem Solving
- Possession of Practical Skills

## Result
Pearson correlation between `Practical_Exposure_Score` and `Professional_Readiness_Score`:

**r = {round(h1_r, 3) if h1_r is not None else 'N/A'}**

## Interpretation
A positive correlation supports H1. This result is used in AHEL as the Student Survey Practical Exposure Benchmark and as hypothesis evidence in the analysis outputs.
"""
(OUT / "student_survey_h1_report.md").write_text(report, encoding="utf-8")
print("Student survey analysis completed.")
