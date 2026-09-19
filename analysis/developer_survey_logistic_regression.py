"""Developer Survey Logistic Regression for AHEL H2.

Outputs:
- outputs/developer_survey_processed.csv
- outputs/developer_logistic_regression_report.md

H2: Higher skill diversity and experience are positively associated with employment status.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "survey_results_public.csv"
if not DATA.exists():
    DATA = ROOT / "data" / "survey_results_public.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SKILL_COLS = [
    "LanguageWorkedWith", "DatabaseWorkedWith", "PlatformWorkedWith",
    "FrameworkWorkedWith", "Methodology", "VersionControl"
]


def split_multi(value):
    if pd.isna(value) or not str(value).strip():
        return []
    return [x.strip() for x in str(value).split(";") if x.strip()]


def parse_years(value):
    text = str(value).lower()
    if "30 or more" in text:
        return 30
    if "less than" in text:
        return 0.5
    nums = [int(x) for x in re.findall(r"\d+", text)]
    if not nums:
        return np.nan
    return float(np.mean(nums[:2])) if len(nums) >= 2 else float(nums[0])


def main():
    df = pd.read_csv(DATA)
    for col in SKILL_COLS:
        if col not in df.columns:
            df[col] = ""

    df["Skill_Count"] = df[SKILL_COLS].apply(lambda row: len(set(sum([split_multi(row[c]) for c in SKILL_COLS], []))), axis=1)
    max_skill_count = max(1, df["Skill_Count"].max())
    df["Skill_Diversity_Index"] = round((df["Skill_Count"] / max_skill_count) * 100, 2)

    df["YearsCoding_Num"] = df.get("YearsCoding", pd.Series([np.nan] * len(df))).apply(parse_years)
    df["YearsCodingProf_Num"] = df.get("YearsCodingProf", pd.Series([np.nan] * len(df))).apply(parse_years)

    employment = df.get("Employment", pd.Series([""] * len(df))).astype(str).str.lower()
    # The provided developer sample contains only employed respondents.
    # Therefore, the binary target is set as full-time employment = 1 and
    # part-time/other employment = 0 so H2 can still test employment status depth.
    df["Employed_Target"] = employment.str.contains("employed full-time", regex=False).astype(int)

    model_df = df[["Skill_Diversity_Index", "Skill_Count", "YearsCoding_Num", "YearsCodingProf_Num", "Employed_Target"]].dropna()
    X = model_df[["Skill_Diversity_Index", "Skill_Count", "YearsCoding_Num", "YearsCodingProf_Num"]]
    y = model_df["Employed_Target"]

    if y.nunique() < 2 or len(model_df) < 20:
        raise ValueError("Not enough valid rows/classes for logistic regression.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)
    cm = confusion_matrix(y_test, pred)
    coef = pd.DataFrame({
        "feature": X.columns,
        "coefficient": pipe.named_steps["model"].coef_[0]
    }).sort_values("coefficient", ascending=False)

    df.to_csv(OUT / "developer_survey_processed.csv", index=False)
    coef.to_csv(OUT / "developer_logistic_coefficients.csv", index=False)

    skill_coef = float(coef.loc[coef["feature"] == "Skill_Diversity_Index", "coefficient"].iloc[0])
    exp_coef = float(coef.loc[coef["feature"] == "YearsCodingProf_Num", "coefficient"].iloc[0])
    h2_text = "partially supported: professional experience is positively associated with full-time employment, while Skill Diversity Index is not positive in this sample" if exp_coef > 0 and skill_coef <= 0 else ("supported" if exp_coef > 0 and skill_coef > 0 else "not supported in this sample")

    report = f"""# Developer Survey Logistic Regression Report

## Hypothesis H2

H2: Higher skill diversity and experience are positively associated with employment status.

## Derived Variable

`Skill_Diversity_Index` was created from the number of distinct technologies listed across language, database, platform, framework, methodology, and version-control fields.

Formula:

`Skill_Diversity_Index = Skill_Count / Maximum_Skill_Count × 100`

## Model

Model used: Logistic Regression  
Target variable: Employed full-time = 1, part-time/other = 0  
Rows used after cleaning: **{len(model_df)}**

## Performance

Accuracy: **{acc:.3f}**  
ROC-AUC: **{auc:.3f}**

Confusion Matrix:

```
{cm}
```

## Coefficients

{coef.to_markdown(index=False)}

## H2 Result

H2 is **{h2_text}**. The dataset is heavily skewed toward full-time employment, so accuracy is high but ROC-AUC and coefficient direction should be interpreted carefully.

## Output Files

- `outputs/developer_survey_processed.csv`
- `outputs/developer_logistic_coefficients.csv`
- `outputs/developer_logistic_regression_report.md`
"""
    (OUT / "developer_logistic_regression_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
