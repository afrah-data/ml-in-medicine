# Healthcare & Insurance Analytics — Machine Learning Portfolio

End-to-end machine learning projects focused on predictive analytics
in healthcare and insurance domains using Python and scikit-learn.

**Author:** Afrah.M  
**Domain:** Healthcare · Insurance · Clinical Analytics  
**Focus:** Classification, imbalanced data, model evaluation, feature interpretation

---

## Projects

### 1. Insurance Cost Classification
**File:** `insurance_classification.ipynb`

Developed a binary classification model to identify high-cost insurance
claimants from demographic and clinical features — a core use case in
actuarial risk assessment and premium pricing.

| | |
|---|---|
| **Dataset** | US Health Insurance Dataset (Kaggle) — 1,338 patients, 7 features |
| **Task** | Binary classification: high-cost vs low-cost patient |
| **Best Model** | Random Forest |
| **Accuracy** | 94.0% |
| **AUC** | 0.9485 |

**What this project covers:**
- Binary target engineering from continuous charge data using median split
- Categorical feature encoding (sex, smoker, region)
- Model comparison: Logistic Regression, Decision Tree, Random Forest
- 5-fold cross-validation for reliable model evaluation
- Confusion matrix, ROC curve, and feature importance analysis
- GridSearchCV hyperparameter tuning with proper holdout methodology
- Data leakage detection and correction
- Business-ready interpretation of results

**Key finding:** Smoking status (coefficient +6.16) is 40x more influential
than any other feature — consistent with actuarial literature on risk stratification.

**Model comparison (5-fold cross-validation):**

| Model | Accuracy | F1 Score | AUC |
|-------|----------|----------|-----|
| Random Forest | 0.9312 | 0.9278 | 0.9487 |
| Logistic Regression | 0.9081 | 0.9081 | 0.9480 |
| Decision Tree | 0.8842 | 0.8843 | 0.8840 |

---

### 2. Hospital 30-Day Readmission Prediction
**File:** `hospital_readmission.ipynb`

Built a readmission risk model on a large real-world clinical dataset,
addressing the class imbalance problem common in healthcare outcomes research.
30-day readmission is a primary quality metric tracked by CMS and hospital systems.

| | |
|---|---|
| **Dataset** | UCI Diabetes 130-US Hospitals — 101,766 patient encounters, 130 hospitals, 1999–2008 |
| **Task** | Binary classification: readmitted within 30 days vs not |
| **Best Model** | Random Forest + SMOTE |
| **AUC** | 0.606 |
| **Key Challenge** | Severe class imbalance — only 11% of patients readmitted |

> **Dataset download:** UCI Machine Learning Repository —
> [Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
> *(diabetic_data.csv — excluded from repo due to file size)*

**What this project covers:**
- Large dataset cleaning: missing values coded as '?', duplicates, high-cardinality columns
- Clinical feature selection from 50+ variables based on domain knowledge
- Class imbalance handling using SMOTE oversampling
- Stratified train/test split to preserve class distribution
- Baseline vs SMOTE model comparison
- Feature importance analysis with clinical interpretation
- Results benchmarked against published literature

**Key finding:** Top predictors — num_lab_procedures, num_medications,
time_in_hospital, and number_inpatient — align with established clinical
risk factors for readmission, validating model interpretability.

**Results:**

| Approach | Accuracy | AUC | Readmission Recall |
|----------|----------|-----|--------------------|
| Random Forest (baseline) | 88.7% | 0.606 | 1% |
| Random Forest + SMOTE | 86.4% | 0.575 | 6% |
| Best published — XGBoost | — | 0.667 | — |

Our results are consistent with published research on the same dataset,
confirming that 30-day readmission is a genuinely difficult prediction
problem due to post-discharge factors not captured in clinical records.

---

## Skills Demonstrated

| Area | Methods |
|------|---------|
| **Classification** | Logistic Regression, Decision Tree, Random Forest |
| **Model Evaluation** | Cross-validation, AUC-ROC, F1, Confusion Matrix, Classification Report |
| **Imbalanced Data** | SMOTE, class_weight balancing, stratified splitting |
| **Hyperparameter Tuning** | GridSearchCV with proper holdout set |
| **Feature Analysis** | Coefficient interpretation, Random Forest feature importance |
| **Data Processing** | Missing value handling, categorical encoding, feature selection |
| **Visualization** | Confusion matrix heatmap, ROC curve, feature importance charts |

---

## Stack

Python 3.10 · Pandas · NumPy · Scikit-learn · Imbalanced-learn · Matplotlib · Seaborn

---

## About

Data Scientist specializing in **healthcare and insurance analytics**.  
MS in Computer Science · 10+ years background in healthcare insurance operations.  
Open to roles in clinical analytics, insurance risk modeling, and healthcare AI.
