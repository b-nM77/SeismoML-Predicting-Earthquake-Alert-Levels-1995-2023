# 🌍 SeismoML: Predicting Earthquake Alert Levels (1995–2023)

This project builds a **machine learning pipeline** to predict earthquake alert levels — *green, yellow, orange,* or *red* — using global seismic data recorded between **1995 and 2023**.  
The goal is to create a **fair and reliable model** that performs well even under **class imbalance**, where lower-severity earthquakes dominate the dataset.

---

## Overview

The dataset includes **1000 global earthquake events**, each with attributes such as:
- Magnitude, depth, and intensity measures (MMI, CDI)  
- Seismic significance score, tsunami indicator, and number of reporting stations  
- Geographic details (latitude, longitude, country, continent)  
- Target variable: **`alert`** level  

The project uses a structured workflow with **scaling, cross-validation, and imbalance correction** to improve predictive stability and fairness across alert categories.

---

## Workflow Summary

| Step | Description |
|------|--------------|
| **1. Data Preprocessing** | Encoding categorical variables, scaling numeric features, and creating a stratified train–test split. |
| **2. Model Selection** | Logistic Regression, Random Forest, and XGBoost used as baseline models. |
| **3. Class-Imbalance Handling** | Compared **Baseline**, **Class Weights**, and **SMOTE** oversampling techniques. |
| **4. Cross-Validation** | Applied **Repeated Stratified K-Fold (5×3)** for reliable performance estimation. |
| **5. Evaluation Metrics** | Accuracy and **Macro-F1** (to treat all alert levels equally). |
| **6. Visualization** | Plotted test-set performance comparison and class distribution charts. |

---

## Results Summary

| Technique | Best Model | Test Accuracy | Test Macro-F1 |
|------------|-------------|----------------|----------------|
| Baseline | Logistic Regression | **0.912** | **0.876** | 
| Class Weights | Logistic Regression | 0.858 | 0.810 | 
| SMOTE | XGBoost | 0.894 | 0.815 | **0.681** |

**Key takeaway:**  
> SMOTE with XGBoost achieved the most **`balanced and stable performance`**

---

## Insights

- Class imbalance significantly affects fairness — without correction, rare high-severity alerts are underpredicted.  
- Synthetic oversampling (SMOTE) and mild regularization improved cross-validation consistency.  
- The final pipeline offers a **robust and interpretable framework** for classifying earthquake alert levels.

---

## Future Improvements

- Experiment with **advanced resampling** (ADASYN, SMOTE-ENN).  
- Include **temporal or regional patterns** to capture location-based trends.

---

- Notebook: [View on Kaggle](https://www.kaggle.com/code/madhurabn/class-imbalance-in-earthquake-prediction)  

---
## Tech Stack

`Python` · `Pandas` · `Scikit-learn` · `Imbalanced-learn` · `XGBoost` · `Seaborn` · `Matplotlib` · `Plotly`

---

> *A complete end-to-end machine learning workflow demonstrating fair and balanced earthquake alert prediction.*

