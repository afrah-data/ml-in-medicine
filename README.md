# Healthcare & Insurance Analytics — Machine Learning Portfolio

End-to-end machine learning projects focused on predictive analytics
in healthcare and insurance domains using Python and scikit-learn.

---

## Projects

### 1. Insurance Cost Classification
**File:** `insurance_classification.ipynb`

Developed a binary classification model to identify high-cost insurance
claimants from demographic and clinical features — a core use case in
actuarial risk assessment and premium pricing.

| | |
|---|---|
| **Dataset** | US Health Insurance Dataset 1,338 patients, 7 features |
| **Best Model** | Random Forest |
| **Accuracy** | 94.0% |
| **AUC** | 0.9485 |

**Approach:**
- Engineered a binary target variable from continuous charge data using median split
- Compared Logistic Regression, Decision Tree, and Random Forest using 5-fold cross-validation
- Identified data leakage during hyperparameter tuning and corrected evaluation methodology
- Produced business-ready interpretation: smoking status accounts for 90% of predictive signal

**Key finding:** A single clinical feature (smoking status, coefficient +6.16) dominates cost prediction — a finding consistent with actuarial literature on risk stratification.

---

### 2. Hospital 30-Day Readmission Prediction
**File:** `hospital_readmission.ipynb`

Built a readmission risk model on a large real-world clinical dataset,
addressing the class imbalance problem common in healthcare outcomes research.
30-day readmission is a primary quality metric tracked by CMS and hospital systems.

| | |
|---|---|
| **Dataset** | UCI Diabetes 130-US Hospitals — 101,766 patient encounters, 130 hospitals, 1999–2008 |
| **Best Model** | Random Forest + SMOTE |
| **AUC** | 0.606 |
| **Readmission Rate** | 11% (severe class imbalance) |

**Approach:**
- Processed and cleaned 100,000+ real clinical records with missing values, duplicates, and high-cardinality features
- Applied SMOTE oversampling to address 89/11 class imbalance
- Selected clinically meaningful features based on domain knowledge
- Benchmarked results against published literature (best reported AUC: 0.667 via XGBoost, PMC 2025)

**Key finding:** Top predictors — number of lab procedures, medications, and prior inpatient visits align with established clinical risk factors for readmission, validating model interpretability.

---

## Technical Skills Demonstrated

| Area | Methods |
|------|---------|
| **Classification** | Logistic Regression, Decision Tree, Random Forest |
| **Model Evaluation** | Cross-validation, AUC-ROC, F1, Confusion Matrix |
| **Imbalanced Data** | SMOTE, class_weight balancing, threshold analysis |
| **Hyperparameter Tuning** | GridSearchCV with proper holdout methodology |
| **Feature Analysis** | Coefficient analysis, Random Forest importance |
| **Data Processing** | Missing value handling, categorical encoding, feature selection |
| **Visualization** | Matplotlib, Seaborn — confusion matrix, ROC curve, feature charts |

---

## Results

| Project | Model | Accuracy | AUC |
|---------|-------|----------|-----|
| Insurance Cost Classification | Random Forest | 94.0% | 0.9485 |
| Hospital Readmission Prediction | Random Forest + SMOTE | 86.4% | 0.606 |

---

## Stack

Python 3.10 · Pandas · NumPy · Scikit-learn · Imbalanced-learn · Matplotlib · Seaborn

---

## About

Data Scientist specializing in **healthcare and insurance analytics**.  
MS in Computer Science with Data Science.  
Open to roles in clinical analytics, insurance risk modeling, and healthcare AI.

