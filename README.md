# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using a Support Vector Machine (SVM) classifier. The dataset is sourced from Kaggle and all preprocessing, analysis, and model training are performed in a Google Colab notebook.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Requirements](#setup--requirements)
- [Workflow](#workflow)
- [Results](#results)

---

## Overview

This project builds a binary classification model to identify whether a credit card transaction is fraudulent (`is_fraud = 1`) or legitimate (`is_fraud = 0`). It covers the full pipeline: data ingestion from Kaggle, exploratory data analysis, preprocessing, feature encoding, model training, and evaluation.

---

## Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/dakshdeepsingh2/fraud-data)

The dataset is fetched directly from Kaggle (via Google Drive in Colab) and contains **555,719 transactions** with **23 columns**, including:

| Column | Description |
|---|---|
| `trans_date_trans_time` | Timestamp of the transaction |
| `cc_num` | Credit card number |
| `merchant` | Merchant name |
| `category` | Transaction category (e.g., personal_care, travel, health_fitness) |
| `amt` | Transaction amount |
| `first`, `last` | Cardholder's first and last name |
| `gender` | Cardholder's gender |
| `street`, `city`, `state`, `zip` | Cardholder's address |
| `lat`, `long` | Cardholder's latitude and longitude |
| `city_pop` | Population of the cardholder's city |
| `job` | Cardholder's occupation |
| `dob` | Cardholder's date of birth |
| `trans_num` | Unique transaction identifier |
| `unix_time` | Transaction timestamp in Unix format |
| `merch_lat`, `merch_long` | Merchant's latitude and longitude |
| `is_fraud` | **Target variable** — 1 if fraudulent, 0 if legitimate |

---

## Project Structure

```
credit-card-fraud-detection/
│
├── Credit_Card_Fraud_Detection.ipynb   # Main Colab notebook
└── README.md                           # Project documentation
```

---

## Setup & Requirements

### Prerequisites

- Python 3.x
- Google Colab (recommended) or a local Jupyter environment
- A Kaggle account and API key (to fetch the dataset)

### Python Libraries

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install locally if needed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Kaggle Dataset Setup

1. Go to your Kaggle account → **Settings** → **API** → **Create New Token**. This downloads a `kaggle.json` file.
2. In Colab, mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Upload the CSV from Kaggle to your Google Drive and update the file path in the notebook accordingly, or use the Kaggle API directly:
   ```bash
   pip install kaggle
   kaggle datasets download -d <dataset-name>
   ```

---

## Workflow

### 1. Data Import
The CSV file is loaded from Google Drive into a Pandas DataFrame. The dataset spans transactions from **June 2020 to December 2020**.

### 2. Exploratory Data Analysis (EDA)
- Inspecting the shape and structure of the data (`df.head()`, `df.info()`)
- Checking for null values and data types
- Visualizing the distribution of fraudulent vs. legitimate transactions using Seaborn

### 3. Data Preprocessing
- Dropping irrelevant or high-cardinality columns (e.g., names, addresses, transaction IDs)
- Encoding categorical features using `LabelEncoder` (e.g., `merchant`, `category`, `gender`, `job`)
- Parsing and engineering date/time features from `trans_date_trans_time` and `dob`

### 4. Feature & Target Split
```python
X = data.drop(columns=["is_fraud"])
Y = data["is_fraud"]
```

### 5. Train-Test Split
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

### 6. Model Training
A **Support Vector Classifier (SVC)** from scikit-learn is trained on the processed dataset:
```python
model = SVC()
model.fit(X_train, Y_train)
```

### 7. Evaluation
```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)
```

---

## Results

| Metric | Value |
|---|---|
| Model | Support Vector Classifier (SVC) |
| Test Accuracy | **99.62%** |

The model achieves a test accuracy of approximately **99.62%** on the held-out test set, demonstrating strong performance in distinguishing fraudulent from legitimate transactions.

> **Note:** Accuracy alone can be misleading on imbalanced datasets. Consider also evaluating with precision, recall, F1-score, and AUC-ROC for a more complete picture of model performance.

---