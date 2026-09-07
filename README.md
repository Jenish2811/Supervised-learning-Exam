# Telco Customer Churn — Supervised Learning

## Overview

This project builds a supervised machine-learning solution to predict **customer churn** from telecommunications customer data.

Customer churn occurs when a customer stops using or cancels a service. Predicting churn can help a telecom business identify high-risk customers early and prioritize retention actions.

The project uses the Telco customer dataset with **7,043 customers and 21 columns**, where `Churn` is the target variable. The dataset contains customer demographics, tenure, subscribed services, contract details, payment method, monthly charges, total charges, and churn status.

## Project Objectives

- Understand the customer churn problem and its business impact.
- Explore and preprocess the Telco customer dataset.
- Handle missing values and categorical variables.
- Address class imbalance using **SMOTE**.
- Train and compare multiple supervised-learning algorithms.
- Evaluate models using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- Select a model using a **Recall-first** strategy because missing a real churner (False Negative) can be costly.
- Save the selected model and preprocessing information for later use.

## Dataset

The project uses the provided Telco customer churn dataset:

- Rows: **7,043**
- Columns: **21**
- Target: `Churn`
- `No Churn`: 5,174 customers (73.46%)
- `Churn`: 1,869 customers (26.54%)

The target is moderately imbalanced, with non-churners forming the majority class.

### Main Features

The dataset includes:

- Customer information: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- Tenure and service information: `tenure`, `PhoneService`, `MultipleLines`, `InternetService`
- Additional services: `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- Contract and billing: `Contract`, `PaperlessBilling`, `PaymentMethod`
- Charges: `MonthlyCharges`, `TotalCharges`
- Target: `Churn`

`customerID` is an identifier and is not used as a predictive feature.

## Machine Learning Workflow

The notebook follows this workflow:

1. **Problem framing and theory**
   - Customer churn
   - Confusion matrix
   - Class imbalance
   - Precision vs. Recall

2. **Exploratory Data Analysis**
   - Dataset inspection
   - Descriptive statistics
   - Churn distribution
   - Univariate and other exploratory analysis

3. **Data preprocessing**
   - Convert `TotalCharges` to numeric.
   - Handle 11 blank `TotalCharges` values using median imputation.
   - Encode binary variables.
   - Encode `gender`.
   - Convert `tenure_group` into an ordered numeric feature.
   - One-hot encode remaining categorical variables.
   - Create derived features such as `tenure_group`, `num_services`, and `AutoPay`.

4. **Train/test split**
   - 80/20 split
   - Stratified by the churn target
   - Random state: `42`

5. **Feature scaling**
   - StandardScaler is applied to:
     - `tenure`
     - `MonthlyCharges`
     - `TotalCharges`
     - `num_services`

6. **Class imbalance handling**
   - SMOTE is applied **only to the training data**.
   - The test set remains untouched.

7. **Model training and evaluation**
   - K-Nearest Neighbours (KNN)
   - Gaussian Naive Bayes
   - Support Vector Machine (SVM)
   - Decision Tree

8. **Model selection**
   - Models are ranked primarily by churn Recall, followed by F1 and ROC-AUC.

9. **Model persistence**
   - The selected model and preprocessing metadata are saved as `churn_model.pkl`.

## Model Results

Evaluation was performed on the untouched test set.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Training Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| GaussianNB | 0.6998 | 0.4630 | **0.8209** | 0.5921 | 0.8099 | 0.0059 |
| Decision Tree | 0.7459 | 0.5140 | 0.7834 | **0.6208** | **0.8144** | 0.0147 |
| KNN | 0.7218 | 0.4849 | 0.7727 | 0.5959 | 0.8120 | 0.0046 |
| SVM | **0.7658** | **0.5495** | 0.6524 | 0.5966 | 0.8014 | 4.1757 |

### Recommended Model

The notebook selects **Gaussian Naive Bayes (GaussianNB)** as the first deployment candidate using a Recall-first ranking.

It achieved:

- **Recall:** 0.8209
- **F1:** 0.5921
- **ROC-AUC:** 0.8099

Recall is prioritized because False Negatives represent customers who are predicted not to churn but actually leave. The notebook also notes that Precision should be monitored because retention teams may have limited campaign capacity.

## Installation

Install the required Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn joblib
```

Alternatively, if a `requirements.txt` file is included in the repository:

```bash
pip install -r requirements.txt
```

## Running the Project

### 1. Clone or download the repository

Place the notebook and dataset in the project directory.

### 2. Check the project files

A typical project layout is:

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

The notebook expects the dataset to be available in the same working environment and loads it with pandas.

## 📸 Images
![Photo](Photo.png)
![Img](img.png)
![Preview](Preview.png)
![Image](Image.png)

### 3. Open the notebook

Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

Then open:

```text
CustomerChurn_SupervisedLearning.ipynb
```

Run the notebook cells from top to bottom.

## Saved Model

The notebook creates:

```text
churn_model.pkl
```

The saved bundle contains:

- Selected model
- StandardScaler
- Columns that require scaling
- Final feature-column list
- Tenure-group ordering
- Binary categorical columns
- Remaining categorical columns
- Target mapping (`No` → `0`, `Yes` → `1`)
- Random state

The notebook also demonstrates loading the saved bundle with `joblib` and generating churn probabilities and predicted labels.

## Business Interpretation

For churn prediction, **Recall is especially important** because a False Negative can represent a customer who leaves without receiving a retention intervention.

At the same time, Precision matters operationally: a model that flags too many customers may cause retention resources to be spent on customers who would not have churned.

The project therefore evaluates both Recall and Precision rather than relying on Accuracy alone.

## Limitations and Future Improvements

Potential next steps identified in the project include:

- Use more recent, out-of-time customer data for validation.
- Tune the probability threshold according to retention-team capacity and business costs.
- Build a real-time scoring API.
- Monitor model performance after deployment.
- Further investigate important churn signals and connect them to targeted retention actions.

## Repository Submission Checklist

The intended repository is:

```text
telco-churn-supervised-learning
```

Recommended contents:

- `CustomerChurn_SupervisedLearning.ipynb` — fully executed notebook
- `churn_model.pkl` — saved model bundle
- `summary_report.md` — project summary report
- `requirements.txt` — Python dependencies
- `README.md` — project overview and run instructions

## Technologies

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- imbalanced-learn / SMOTE
- joblib
- Jupyter Notebook
