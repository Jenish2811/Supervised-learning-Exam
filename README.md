# 📊 Telco Customer Churn Prediction & Retention System

An end-to-end Machine Learning and Customer Analytics pipeline designed to predict customer churn in the telecommunications sector (e.g., Jio, Airtel, Vodafone Idea, AT&T). Using exploratory data analysis, class imbalance handling via SMOTE, robust feature engineering, and multiple supervised classification algorithms (**K-Nearest Neighbors, Gaussian Naive Bayes, Support Vector Machines, and Decision Trees**), this project provides actionable churn risk scoring to optimize Customer Retention and maximize Customer Lifetime Value (CLV).

---

## 📑 Table of Contents
1. [Business Problem & Motivation](#-business-problem--motivation)
2. [Dataset Overview](#-dataset-overview)
3. [Key Exploratory Data Analysis (EDA) Insights](#-key-exploratory-data-analysis-eda-insights)
4. [Data Preprocessing & Feature Engineering](#-data-preprocessing--feature-engineering)
5. [Class Imbalance & SMOTE](#-class-imbalance--smote)
6. [Models & Evaluation Methodology](#-models--evaluation-methodology)
7. [Benchmark Model Comparison](#-benchmark-model-comparison)
8. [Decision Tree Analysis & Feature Importances](#-decision-tree-analysis--feature-importances)
9. [Project Structure](#-project-structure)
10. [Installation & Setup](#-installation--setup)
11. [Usage & Inference](#-usage--inference)
12. [Business Recommendations & Action Plan](#-business-recommendations--action-plan)

---

## 🧠 Business Problem & Motivation

### What is Customer Churn?
Customer churn occurs when a subscriber cancels their contract or discontinues services with a telecom operator. 

### Why Churn Prediction is Vital:
* **Customer Acquisition Cost (CAC) vs. Customer Lifetime Value (CLV):** Acquiring a new telecom customer costs **5x to 7x more** than retaining an existing one. Proactive retention saves marketing capital and protects baseline recurring revenue.
* **Cost Asymmetry in Confusion Matrix:**
  * **False Positive (FP):** Model predicts churn, but customer would have stayed. Cost: Small expense of a promotional retention discount or call.
  * **False Negative (FN):** Model predicts no churn, but customer leaves. Cost: Complete loss of remaining CLV and market share to competitors.
  * **Recall-Oriented Strategy:** Minimizing False Negatives is paramount in telecom churn management. High recall ensures high-risk customers receive intervention before porting out.

---

## 📂 Dataset Overview

The dataset contains profile and subscription details for **7,043 telecom customers** across **21 features**:
* **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
* **Account Information:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
* **Subscribed Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
* **Target:** `Churn` (`Yes` / `No`)

### Target Distribution
* **Non-Churners (`No`):** 5,174 customers (~73.46%)
* **Churners (`Yes`):** 1,869 customers (~26.54%)
* **Imbalance Ratio:** ~2.77 : 1 (Moderately imbalanced classification problem)

---

## 🔍 Key Exploratory Data Analysis (EDA) Insights

1. **Contract Type Impact:**
   * **Month-to-month contracts** exhibit the highest churn rate (~**42.71%**), whereas **Two-year contracts** exhibit the lowest churn (~**2.83%**).
   * Long-term contracts lock in customer loyalty and drastically reduce churn propensity.
2. **Tenure Dynamics:**
   * Customers in the **0–12 month tenure bucket** have a staggering **47.44% churn rate**.
   * Churn decreases steeply as tenure increases: 13–24 months (~28.71%), 25–48 months (~20.39%), and 49–72 months (~9.51%).
3. **Monthly Charges & Fiber Optic Risk:**
   * Churning customers pay significantly higher monthly charges on average (**$74.44**) compared to non-churners (**$61.27**).
   * Fiber optic internet users churn at high rates due to higher pricing and service reliability expectations.
4. **Payment Methods:**
   * Customers paying via **Electronic check** demonstrate the highest churn rate, whereas automated bank transfers and credit cards show the lowest churn.
5. **Highest-Risk Customer Persona:**
   * *Month-to-month contract + Short tenure (<= 12 months) + High monthly charges (>= median)* yields a peak churn rate of **~69.73%**.

---

## 🔧 Data Preprocessing & Feature Engineering

1. **Data Cleaning:**
   * Converted `TotalCharges` from string/object to float64, imputing 11 blank whitespace values with the column median.
   * Dropped non-informative identifier: `customerID`.
2. **Feature Engineering:**
   * `tenure_group`: Ordinal binning of tenure into 4 cohorts (`New: 0-12`, `Mid: 13-36`, `Senior: 37-60`, `Loyal: 61-72`).
   * `num_services`: Sum of value-added active services subscribed (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`).
   * `AutoPay`: Binary indicator (1 if payment method contains 'automatic', else 0).
3. **Encoding:**
   * Binary columns (`Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `gender`) mapped to `0` and `1`.
   * Ordinal encoding for `tenure_group`.
   * Multi-class nominal categoricals (`Contract`, `InternetService`, `PaymentMethod`, etc.) one-hot encoded (`pd.get_dummies`), yielding a total of **43 feature columns**.
4. **Standard Scaling:**
   * Continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`, `num_services`) standardized using `StandardScaler` fitted exclusively on training data.

---

## ⚖️ Class Imbalance & SMOTE

* **Technique:** Synthetic Minority Over-sampling Technique (`SMOTE`) applied with `random_state=42`.
* **Data Integrity Protocol:** SMOTE was fitted **strictly on the training partition** (`X_train_scaled`, `y_train`) to prevent data leakage:
  * Before SMOTE: `4,139 No Churn` vs. `1,495 Churn` (Total: 5,634)
  * After SMOTE: `4,139 No Churn` vs. `4,139 Churn` (Total: 8,278 balanced samples)
  * Test Set: Preserved in its original natural distribution (`1,409` samples: 1,035 No Churn, 374 Churn).

---

## 🤖 Models & Evaluation Methodology

Four core supervised learning algorithms were benchmarked:
1. **K-Nearest Neighbors (KNN):** Distance-based non-parametric classifier. Sensitive to feature scale; tuned over different `k` values.
2. **Gaussian Naive Bayes (GaussianNB):** Probabilistic model leveraging Bayes' theorem with Gaussian continuous likelihood assumptions. Fast, robust baseline with exceptional recall on minority classes.
3. **Support Vector Machine (SVM):** Maximum-margin hyper-plane classifier with RBF kernel to capture non-linear decision boundaries.
4. **Decision Tree Classifier:** Highly interpretable recursive partitioning tree evaluated with Gini impurity and depth pruning (`max_depth` cross-validation).

### Evaluation Metrics
Given the business asymmetry of churn:
* **Recall (Churn):** Primary metric (captures the proportion of actual churners identified).
* **Precision (Churn):** Measures retention campaign efficiency (avoiding unnecessary incentives).
* **F1-Score:** Harmonic mean balancing precision and recall.
* **ROC-AUC:** Discriminative ability across all classification thresholds.

---

## 📈 Benchmark Model Comparison

### Performance Summary on Test Set (1,409 samples)

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (Tuned)** | **75.4%** | **0.515** | **0.783** | **0.621** | **0.814** |
| **KNN (k=5 / Tuned)** | 73.6% | 0.485 | 0.773 | 0.596 | 0.812 |
| **Gaussian Naive Bayes** | 70.1% | 0.463 | **0.821** | 0.592 | 0.810 |
| **SVM (RBF Kernel)** | 77.2% | 0.547 | 0.652 | 0.595 | 0.801 |

### Key Observations from Evaluation Plots:
* **ROC Curves:** All four models demonstrate competitive discriminative power with AUCs clustering between **0.801 and 0.814**, led by the Decision Tree (**0.814**).
* **Precision vs. Recall Tradeoff:** 
  * **GaussianNB** achieved the highest raw Recall (**82.1%**), making it an excellent candidate for aggressive, wide-funnel retention campaigns.
  * **Decision Tree** offered the most balanced operating performance with **78.3% Recall**, **51.5% Precision**, and the top **F1-score (0.621)**.
  * **SVM** favored Precision (**54.7%**) at the expense of Recall (**65.2%**), ideal for capacity-constrained call centers.

---

## 🌲 Decision Tree Analysis & Feature Importances

### Hyperparameter Tuning (`max_depth` vs. CV F1)
* Tree depth was evaluated across depths `[3, 4, 5, 6, 7, 8, None]`.
* An unconstrained tree (`max_depth=None`) severely overfits the training data, dropping CV F1 below **0.50**.
* Optimal performance peaked at **`max_depth=5` (and 3)** with a cross-validated F1 score of **~0.62**, balancing bias and variance.

### Top Splitting Features (Root & Early Nodes):
1. **`Contract_Month-to-month <= 0.5`**: The single most powerful split in the entire tree.
2. **`InternetService_Fiber optic <= 0.5`**: Primary pricing/infrastructure risk indicator.
3. **`tenure`**: New accounts (< 12 months) split sharply toward churn leaves.
4. **`MonthlyCharges` & `TotalCharges`**: High-cost contracts accelerate churn.
5. **`PaymentMethod_Electronic check` & `TechSupport_No`**: Friction points in billing and customer care.

---

## 📁 Project Structure

```text
├── dataset.csv                 # Raw Telco Customer Churn dataset (7,043 rows, 21 columns)
├── notebook.ipynb              # Jupyter Notebook with complete EDA, preprocessing, and model pipelines
├── requirements.txt            # Python dependencies
├── model.joblib / model.pkl    # Serialized production pipeline (Model, Scaler, Column Metadata)
└── README.md                   # Detailed project documentation and business guidelines
├── README.md
└── images/
    my-project/
├── README.md
    └── img.png
     └── Photo.png
      └── Image.png
       └── Preview.png
```

---

## 💻 Installation & Setup

### Prerequisites
* Python 3.9+ 
* Virtual environment (recommended)

### 1. Clone the repository / setup directory
```bash
git clone https://github.com/your-username/telco-churn-prediction.git
cd telco-churn-prediction
```

### 2. Create and activate a virtual environment
```bash
# On Linux / macOS
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install required dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage & Inference

### 1. Run the Jupyter Notebook
To reproduce all figures, exploratory analysis, and evaluation benchmarks:
```bash
jupyter notebook notebook.ipynb
```

### 2. Programmatic Model Inference with Joblib
```python
import joblib
import pandas as pd

# Load saved pipeline dictionary
artifact = joblib.load("model.joblib")
model = artifact["model"]
scaler = artifact["scaler"]
scale_cols = artifact["scale_cols"]
feature_columns = artifact["feature_columns"]

# Example: Prepare customer input dataframe (43 encoded columns)
# customer_df = pd.DataFrame([...])
# customer_df[scale_cols] = scaler.transform(customer_df[scale_cols])
# churn_prediction = model.predict(customer_df[feature_columns])
# churn_probability = model.predict_proba(customer_df[feature_columns])[:, 1]
```

---


## 📸 Images
![Images](Image.png)
![Img](img.png)
![Photo](Photo.png)
![Preview](Preview.png)


## 💡 Business Recommendations & Action Plan

1. **Incentivize Annual & Bi-Annual Contract Migration:**
   * Month-to-month contracts have a **42.7% churn rate**. Offer structured discounts or loyalty perks (e.g., 2 free months of streaming) to transition high-value month-to-month users into 1-year commitments.
2. **Onboarding & First-Year Nurturing Campaign:**
   * Over **47%** of first-year subscribers churn. Create targeted onboarding check-ins at Day 30, 90, and 180 to resolve setup and service friction before users consider competitor switching.
3. **Fiber Optic Service Quality & Value Bundling:**
   * Address high churn among fiber users by bundling free `TechSupport` and `OnlineSecurity`. Customers with active tech support exhibit significantly higher retention.
4. **Deprecate Electronic Check / Promote AutoPay:**
   * Introduce a $2–$5 monthly bill credit for setting up automated bank transfers or credit card payments, mitigating payment-failure and manual billing churn.
