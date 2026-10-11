# Credit Card Fraud Detection

An end-to-end machine learning project for detecting fraudulent credit card transactions under severe class imbalance.

The project demonstrates data cleaning, exploratory data analysis, imbalance handling, supervised machine learning, threshold optimization, and business-oriented fraud detection evaluation.

## Project Objective

The objective is to build a fraud detection pipeline capable of identifying fraudulent transactions while balancing:

* Fraud detection rate
* False positives
* False negatives
* Precision
* Recall
* F1-score
* ROC-AUC
* Business risk

Rather than relying only on accuracy, the project focuses on fraud-specific evaluation metrics and decision-threshold optimization.

## Dataset

This project uses a synthetic credit card transaction dataset created for educational and portfolio purposes.

The dataset contains:

* `Time` — transaction time representation
* `V1`–`V28` — anonymized transaction features
* `Amount` — transaction amount
* `Class` — transaction label

Where:

* `0` = Legitimate transaction
* `1` = Fraudulent transaction

The dataset contains 5,000 transactions with a deliberately imbalanced class distribution.

### Dataset Note

The dataset is synthetic and is not the original Kaggle Credit Card Fraud Detection dataset.

Therefore, the model performance obtained in this project should not be interpreted as real-world banking or financial fraud detection performance.

This project uses a synthetic credit card transaction dataset created for educational and portfolio purposes.

The dataset contains transaction features such as Time, Amount, V1-V28,
and Class, where Class represents legitimate and fraudulent transactions.

The dataset is intentionally designed to demonstrate:

- Class imbalance
- Fraud detection
- Sampling techniques
- Model comparison
- Precision/Recall trade-offs
- Decision threshold optimization
- Business-oriented fraud detection analysis

The results should therefore not be interpreted as real-world banking or
financial fraud detection performance.

## Project Workflow

```text
Dataset
   ↓
Data Validation
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Class Imbalance Analysis
   ↓
Train/Test Split
   ↓
Feature Scaling
   ↓
Sampling Techniques
   ├── Original
   ├── Random Under-Sampling
   ├── Random Over-Sampling
   └── SMOTE
   ↓
Model Training
   ├── Random Forest
   └── XGBoost
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
False Negative Analysis
   ↓
Decision Threshold Analysis
   ↓
Business Risk Evaluation
   ↓
Final Model Selection
```

## Key Features

### 1. Data Validation

The pipeline checks:

* Dataset structure
* Missing values
* Duplicate records
* Data types
* Class distribution
* Numerical features

### 2. Exploratory Data Analysis

The project analyzes:

* Fraud vs legitimate transactions
* Transaction amounts
* Transaction timing
* Feature distributions
* Correlations
* Class imbalance

EDA outputs are stored in:

```text
outputs/eda/
```

### 3. Class Imbalance Handling

Fraud datasets are typically highly imbalanced.

The project compares:

* Original imbalanced dataset
* Random Under-Sampling
* Random Over-Sampling
* SMOTE

Sampling is performed only on the training data to prevent test-set contamination.

Sampling outputs are stored in:

```text
outputs/sampling/
```

### 4. Machine Learning Models

Two classification algorithms are evaluated:

* Random Forest
* XGBoost

### 5. Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix
* True Negatives
* False Positives
* False Negatives
* True Positives

Special attention is given to **Recall**, because missing fraudulent transactions can represent significant business risk.

### 6. Decision Threshold Optimization

The default classification threshold is not always optimal for fraud detection.

The project evaluates multiple probability thresholds and analyzes their effect on:

* Precision
* Recall
* F1-score
* False positives
* False negatives

This allows the model to be aligned with different business risk preferences.

### 7. Business Analysis

The project translates model predictions into business-oriented decisions by examining:

* Fraud detection capability
* False-positive impact
* False-negative risk
* Precision/recall trade-offs
* Recommended analytical threshold

Business outputs are stored in:

```text
outputs/business/
```

## Model Result

During the project run, the strongest model candidate was:

```text
Model:       XGBoost
Sampling:    SMOTE
Precision:   0.9048
Recall:      0.9500
F1-score:    0.9268
ROC-AUC:     0.9997
```

The analytical threshold selected by the pipeline was:

```text
0.20
```

These results are specific to the synthetic dataset used in this project and should not be generalized to real-world fraud detection systems.

## Project Structure

```text
Credit_Card_Fraud_Detection/
│
├── data/
│   └── creditcard.csv
│
├── models/
│
├── outputs/
│   ├── eda/
│   ├── sampling/
│   ├── models/
│   └── business/
│
├── src/
│   └── credit_card_fraud_detection.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* XGBoost
* Joblib

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd Credit_Card_Fraud_Detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Run the main pipeline:

```bash
python src/credit_card_fraud_detection.py
```

The pipeline generates analytical outputs, model evaluation results, threshold analysis, and validation reports inside the `outputs/` directory.

## Important Machine Learning Practices

The project follows several important ML practices:

* Stratified train/test splitting
* Scaling based only on training data
* Sampling only the training data
* Separate test-set evaluation
* Precision/Recall analysis for imbalanced classification
* False-negative analysis
* Decision-threshold optimization
* Model comparison using multiple metrics

## Business Interpretation

Fraud detection is not simply an accuracy problem.

A model can achieve high accuracy while still failing to identify fraudulent transactions when legitimate transactions greatly outnumber fraudulent transactions.

Therefore, this project emphasizes:

```text
Recall → How much fraud is detected?
Precision → How many flagged transactions are actually fraud?
F1-score → Balance between precision and recall
False Negatives → Fraud that the system failed to detect
False Positives → Legitimate transactions incorrectly flagged
```

The optimal threshold can therefore depend on the business cost associated with false positives and false negatives.

## Future Improvements

Possible future enhancements include:

* Hyperparameter optimization
* Probability calibration
* Cost-sensitive learning
* Advanced anomaly detection
* LightGBM comparison
* Explainable AI using SHAP
* Real-time fraud scoring API
* Model monitoring
* Data drift detection
* Production deployment
* Real-world benchmark dataset evaluation

## Author

**Aman Singh**

Data Science | Data Analytics | Machine Learning

---

*This project was developed for educational, portfolio, and machine-learning practice purposes.*
