# Fraud Transaction Detection with Machine Learning
**By Abhishek | April, 2026**

---

## Introduction
Financial fraud is a growing concern in digital payment systems. With millions of transactions occurring daily, manually identifying fraudulent activity is nearly impossible. This project focuses on building an automated fraud detection system by combining classical machine learning techniques with robust feature engineering to accurately flag suspicious transactions.

## Objective
- Build a Machine Learning pipeline to classify transactions as **fraudulent or legitimate**
- Understand the role of **feature engineering** and **multicollinearity reduction (VIF)** in improving model quality
- Compare the performance of **Decision Tree** and **Random Forest** classifiers
- Explore model evaluation metrics suited for **highly imbalanced datasets**

---

## Data Overview
I used the **PaySim1** dataset from Kaggle, a simulated mobile money transaction dataset inspired by real financial logs.

- **Total Records:** 6,362,620 transactions
- **Features:** 11 columns (step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud)
- **Class Distribution:**
  - Legit: 6,354,407 → **99.87%**
  - Fraud: 8,213 → **0.13%** *(severely imbalanced)*

Sample data:

| step | type | amount | nameOrig | isFraud |
|------|------|--------|----------|---------|
| 1 | PAYMENT | 9839.64 | C1231006815 | 0 |
| 1 | TRANSFER | 181.00 | C1305486145 | 1 |

---

## Model Structure
The fraud detection pipeline is split into two concerns:

- **Feature Extraction** — engineers meaningful numeric signals from raw transaction data
- **Classification** — learns patterns from those signals to predict fraud

### Models Used
| Model | Description |
|-------|-------------|
| Decision Tree | Fast, interpretable baseline classifier |
| Random Forest | Ensemble of 100 trees; more robust and accurate |

---

## Data Engineering

Cleaning steps applied:
- Label encoding of `type`, `nameOrig`, `nameDest`
- Normalization of `amount` using **StandardScaler**
- Removal of redundant/high-VIF columns

### Multicollinearity Check (VIF)
Before feature engineering, `oldbalanceOrg` and `newbalanceOrig` had extreme VIF scores (576 and 582), indicating severe multicollinearity.

**Fix — derived features replacing correlated pairs:**
- `Actual_amount_orig` = oldbalanceOrg − newbalanceOrig
- `Actual_amount_dest` = oldbalanceDest − newbalanceDest
- `TransactionPath` = nameOrig + nameDest

**VIF After Feature Engineering:**

| Variable | VIF |
|----------|-----|
| type | 2.69 |
| amount | 3.82 |
| Actual_amount_orig | 1.31 |
| Actual_amount_dest | 3.75 |
| TransactionPath | 2.68 |

All VIF values dropped below 5 — multicollinearity resolved ✅

---

## Model Training

```python
# Train/Test Split (70/30)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
# Train: 4,453,834 | Test: 1,908,786

dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
```

---

## Model Evaluation

### Decision Tree
| | Precision | Recall | F1-Score |
|--|-----------|--------|----------|
| Legit (0) | 1.00 | 1.00 | 1.00 |
| Fraud (1) | 0.70 | 0.71 | 0.70 |

`TP=1717 | FP=731 | TN=1905620 | FN=718`

### Random Forest
| | Precision | Recall | F1-Score |
|--|-----------|--------|----------|
| Legit (0) | 1.00 | 1.00 | 1.00 |
| **Fraud (1)** | **0.97** | **0.70** | **0.81** |

`TP=1713 | FP=60 | TN=1906291 | FN=722`

> ✅ Random Forest reduced **false positives from 731 → 60** (91.8% improvement) — critical since wrongly flagging legitimate users is costly.

---

## Feature Importance (Random Forest)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Actual_amount_orig | ~0.39 |
| 2 | Actual_amount_dest | ~0.37 |
| 3 | NormalizedAmount | ~0.13 |
| 4 | TransactionPath | ~0.07 |
| 5 | type | ~0.04 |

Balance change at origin and destination are the strongest fraud signals.

---

## Conclusion
This project built an end-to-end fraud detection system on over 6 million transactions. Random Forest achieved **0.97 precision** on fraud cases with a macro-avg F1 of 0.91, significantly outperforming the Decision Tree baseline. VIF-based feature engineering was key to improving model stability.

The model performs well, though it lacks semantic understanding of account behavior over time — patterns such as a user suddenly making large transfers after months of inactivity are not captured.

---

## Future Work
- Apply **SMOTE** to better handle class imbalance
- Explore **XGBoost / LightGBM** for faster ensemble learning
- Add **time-series features** to capture behavioral patterns across transaction sequences
- Deploy as a **REST API** for real-time fraud scoring

---

## How to Run

```bash
git clone https://github.com/your-username/fraud-transaction-detection.git
cd fraud-transaction-detection
pip install -r requirements.txt
# Open fraud_transaction_detection.ipynb in Google Colab
```

## Dataset
[PaySim1 on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) — License: CC BY-SA 4.0

## References
[1] E. Lopez-Rojas, A. Elmir, S. Axelsson. PaySim: A financial mobile money simulator. 2016.  
[2] Pedregosa et al. Scikit-learn: Machine Learning in Python. JMLR 12, 2011.  
[3] Breiman, L. Random Forests. *Machine Learning*, 45(1), 5–32, 2001.
