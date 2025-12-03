# Payback-Prediction-Credit-Risk-Machine-Learning
This project builds end-to-end ML pipeline: EDA, cleaning, feature engineering, model training and tuning, evaluation, ensemble, test prediction.

An end-to-end machine learning project to predict **loan repayment vs. default risk** from borrower and loan attributes.  
This work covers the full ML workflow: **EDA → preprocessing → feature engineering → model training & tuning → ensemble → calibration → prediction**.

**Kaggle Notebook:** https://www.kaggle.com/code/xavierbenedict/mastering-loan-default-prediction-eda-0-92-auc  

## Project Highlights
- Built a complete credit-risk ML pipeline from raw data to final predictions.
- Trained and compared strong models:
  - Logistic Regression (baseline)
  - Random Forest
  - Neural Network (MLP)
  - XGBoost
  - LightGBM
  - **Stacking Classifier (ensemble)**
- Evaluated both **ranking performance (ROC-AUC, PR-AUC)** and **probability quality (Brier Score)**.
- Final models achieve ~**0.92 ROC-AUC** with well-calibrated probabilities.

## Problem Statement
Credit risk modeling is a key function in finance.  
Given historical loan data, the goal is to **estimate the probability that a borrower will repay a loan**, enabling better approval decisions and risk-based pricing.

- Target:
  - `Repay = 1`
  - `Default = 0`
 ## Evaluation Metrics
- **ROC-AUC** (primary ranking metric)
- Precision / Recall / F1
- Confusion Matrix
- **Average Precision (PR-AUC)**
- **Brier Score** for probability calibration

## Model Comparison (Validation Set)

| Model | ROC-AUC | Brier Score | Average Precision (AUC-PR) |
|---|---:|---:|---:|
| **Stacking Classifier** | **0.920423** | 0.073075 | **0.974408** |
| **LightGBM** | 0.920399 | **0.072187** | 0.974378 |
| XGBoost | 0.918842 | 0.108663 | 0.973925 |
| Random Forest | 0.912710 | 0.104030 | 0.971235 |
| Neural Network | 0.911271 | 0.075587 | 0.971039 |
| Logistic Regression | 0.793664 | 0.128313 | 0.934097 |

### Key Takeaways
- **Stacking achieved the best ROC-AUC (0.9204)** and best PR-AUC → strongest overall classifier.
- **LightGBM matched Stacking on ROC-AUC**, but had the **lowest Brier Score (0.0722)** → best-calibrated probabilities.
- Boosting methods (LightGBM/XGBoost) outperform RF/NN baselines here.
- Logistic Regression provides a useful baseline but underfits nonlinear patterns.

## Tech Stack
Python, Pandas, NumPy, Matplotlib/Seaborn, Scikit-learn,  
XGBoost, CatBoost, LightGBM, Jupyter Notebook

## Contact

Thanh Xuyen, Nguyen

LinkedIn: [xuyen-thanh-nguyen-0518](https://www.linkedin.com/in/xuyen-thanh-nguyen-0518/)

Email: thanhxuyen.nguyen@outlook.com
