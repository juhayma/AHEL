# AHEL Model Training Report

## Exported Model Files

- `models/logistic_regression_model.pkl`
- `models/tfidf_vectorizer.pkl`
- `models/label_encoder.pkl`
- `models/specialization_weights.json`
- `models/skill_clusters.json`

## Logistic Regression Model

Purpose: Tests H2 by modelling the relationship between skill diversity, experience depth, and employment status.

Rows used: **2157**
Features:
- Skill_Diversity_Index
- Skill_Count
- YearsCoding_Num
- YearsCodingProf_Num

Accuracy: **0.470**
ROC-AUC: **0.761**

Confusion matrix:

```text
[[229 284]
 [  2  25]]
```

Classification report:

```text
              precision    recall  f1-score   support

   full_time       0.99      0.45      0.62       513
       other       0.08      0.93      0.15        27

    accuracy                           0.47       540
   macro avg       0.54      0.69      0.38       540
weighted avg       0.95      0.47      0.59       540

```

## TF-IDF Vectorizer

Purpose: NLP preprocessing and feature extraction from job titles/descriptions.

The vectorizer uses lowercase normalization, stop-word removal, unigrams/bigrams, and a restricted technical-token pattern.
