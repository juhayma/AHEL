"""Run all AHEL analysis files and generate combined hypothesis summary."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    ROOT / "analysis" / "job_market_analysis.py",
    ROOT / "analysis" / "developer_survey_logistic_regression.py",
    ROOT / "analysis" / "student_survey_analysis.py",
]

for script in SCRIPTS:
    print(f"\nRunning {script.name}...")
    subprocess.run([sys.executable, str(script)], check=True)

out = ROOT / "outputs" / "hypothesis_results_summary.md"
parts = ["# AHEL Hypothesis Results Summary\n"]
for name in ["student_survey_h1_report.md", "developer_logistic_regression_report.md", "job_market_summary.md"]:
    path = ROOT / "outputs" / name
    if path.exists():
        parts.append(path.read_text(encoding="utf-8"))
        parts.append("\n---\n")
out.write_text("\n".join(parts), encoding="utf-8")
print(f"\nCombined summary written to {out}")
