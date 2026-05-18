# 📡 Telco Customer Churn Prediction

An end-to-end machine learning pipeline that predicts customer churn for a telecommunications company. The project covers the full ML lifecycle — from exploratory data analysis through feature engineering, model selection, and interpretability — to identify at-risk customers before they leave.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Key Insights](#key-insights)
- [Pipeline Stages](#pipeline-stages)
- [Model Performance](#model-performance)
- [Feature Importance](#feature-importance)
- [Installation & Usage](#installation--usage)
- [Results](#results)
- [Conclusion](#conclusion)

---

## Overview

Customer churn is one of the most critical challenges in the telecom industry. This project builds a predictive pipeline that:

1. **Explores** the data through univariate, bivariate, and multivariate analysis
2. **Engineers** meaningful features (service count, tenure buckets)
3. **Trains & tunes** 6 different classification models with cross-validation
4. **Interprets** predictions using SHAP feature importance
5. **Saves** trained models for future deployment

---

## Project Structure

```
Telco Customer Churn/
│
├── main.py                         # End-to-end pipeline entry point
├── requirements.txt                # Python dependencies
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset (7,043 customers)
│
├── src/
│   ├── __init__.py
│   ├── EDA.py                      # Exploratory Data Analysis plots
│   ├── features_engineering.py     # Feature creation & encoding
│   ├── modeling.py                 # Model training, evaluation, SHAP
│   └── utils.py                    # Save / load model utilities
│
├── notebook/
│   └── telco-customer-churn.ipynb  # Interactive Jupyter notebook
│
├── models/                         # Saved trained models (.pkl)
│
└── outputs/                        # EDA plots & result charts
    ├── categorical_bivariate.jpg
    ├── numerical_univariate.png
    ├── numeric_feature_churn_status.png
    ├── violinplot.png
    ├── correlationmatrix.jpg
    ├── Pairplot.jpg
    ├── model_val_accuracy.png
    └── model_accuracy_comparison.png
```

---

## Dataset

| Property | Value |
|---|---|
| **Source** | [IBM Sample – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Rows** | 7,043 customers |
| **Columns** | 21 features |
| **Target** | `Churn` (Yes / No) |
| **Class Balance** | ~26.5% churned, ~73.5% retained |

### Features at a Glance

| Category | Features |
|---|---|
| **Demographics** | Gender, SeniorCitizen, Partner, Dependents |
| **Account** | Tenure, Contract, PaperlessBilling, PaymentMethod |
| **Services** | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| **Charges** | MonthlyCharges, TotalCharges |

---

## Key Insights

### 🔍 Customer Profile Most Likely to Churn

- **New customers** with less than 12 months tenure
- Subscribed to **Fiber optic** internet service
- On a **Month-to-Month** contract
- Paying via **Electronic Check** (churn rate ≈ 45.3% — ~3× other methods)
- **No add-on services** (no online security, tech support, or device protection)
- **No partner** and **no dependents**

### 🔑 Most Important Features

| Rank | Feature | Impact |
|---|---|---|
| 1 | **Tenure** | Single strongest predictor — shorter tenure = higher risk |
| 2 | **Contract type** | Two-year contracts are the strongest protective factor |
| 3 | **Monthly charges** | Higher charges increase churn, especially for new customers |
| 4 | **Internet service** | Fiber optic customers churn at nearly 2× the rate of DSL |
| 5 | **Payment method** | Electronic check users churn at ~3× the rate of other methods |

---

## Pipeline Stages

### 1. Data Preprocessing
- Convert `TotalCharges` to numeric (11 missing values filled with mean)
- Drop `customerID` (non-predictive identifier)
- Map `SeniorCitizen` from 0/1 to categorical yes/no

### 2. Exploratory Data Analysis (`src/EDA.py`)
- **Univariate**: Bar plots for categorical features, histogram + KDE for numerical
- **Bivariate**: Crosstab bar plots, box plots, violin plots (MonthlyCharges × Contract × Churn)
- **Multivariate**: Correlation matrix, pair plot

### 3. Feature Engineering (`src/features_engineering.py`)
- `Num_of_services` — count of subscribed add-on services (security, backup, etc.)
- `tenure_bucket` — binned tenure into 7 groups (0–6m, 7–12m, …, 60m+)
- **Binary encoding** for binary features (LabelEncoder)
- **One-Hot encoding** for multi-class categorical features (`pd.get_dummies`)

### 4. Data Splitting
- **60%** Training  /  **32%** Validation  /  **8%** Test
- Stratified split to preserve class balance

### 5. Modeling (`src/modeling.py`)
- 5-Fold Cross-Validation for generalization assessment
- Hyperparameter tuning via GridSearchCV / RandomizedSearchCV
- Models trained: Logistic Regression, KNN, Random Forest, XGBoost, CatBoost, GBM

### 6. Interpretability
- SHAP (SHapley Additive exPlanations) feature importance for the best model

### 7. Model Persistence (`src/utils.py`)
- All trained models saved as `.pkl` files via `joblib`
- Results DataFrame saved alongside models

---

## Model Performance

| Model | CV Accuracy | Validation Accuracy |
|---|---|---|
| **XGBoost** | **0.806** | **0.810** |
| Logistic Regression | 0.803 | 0.810 |
| GBM | 0.804 | 0.803 |
| Random Forest | 0.799 | 0.801 |
| CatBoost | 0.792 | 0.798 |
| KNN Classifier | 0.780 | 0.777 |

> **Best Model: XGBoost** — achieved the highest AUC of **0.81** with strong generalization (minimal gap between CV and validation accuracy).

---

## Feature Importance

Based on SHAP analysis of the XGBoost model:

- **Tenure** is the most important feature for predicting churn
- **Two-year contracts** significantly reduce churn prediction
- Having **no internet service** or **low monthly charges** also reduce churn risk
- **Fiber optic** internet and **electronic check** payment push predictions toward churn

---

## Installation & Usage

### Prerequisites
- Python 3.9+

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/telco-customer-churn.git
cd telco-customer-churn

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Full pipeline (EDA + Training + Save)
python main.py

# Skip EDA plots (faster)
python main.py --skip_eda

# Custom directories
python main.py --data_dir data/ --models_dir models/ --output_dir outputs/
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `data/` | Directory containing the CSV file |
| `--models_dir` | `models/` | Directory to save trained models |
| `--output_dir` | `outputs/` | Directory to save plots and results |
| `--skip_eda` | `False` | Skip EDA visualizations |

---

## Results

The pipeline generates the following outputs:

### EDA Visualizations (in `outputs/`)
- Categorical feature distributions with churn breakdown
- Numerical feature distributions (histogram + KDE)
- Box plots and violin plots comparing churned vs. retained
- Correlation matrix and pair plots

### Model Comparison Charts (in `outputs/`)
- Validation accuracy bar chart across all models
- CV accuracy vs. validation accuracy line chart

### Saved Models (in `models/`)
- 6 trained model files (`.pkl`)
- Results summary DataFrame

---

## Conclusion

**Churn is not random.** It follows clear, detectable patterns. With the right model, a telecom company can meaningfully reduce churn and protect its revenue. This pipeline demonstrates that even with a relatively simple feature set, models like XGBoost can identify at-risk customers with over 80% accuracy — enabling proactive retention strategies.

---

## 🛠 Tech Stack

- **Python** — pandas, NumPy, scikit-learn
- **Models** — XGBoost, CatBoost, GradientBoosting, RandomForest, LogisticRegression, KNN
- **Visualization** — matplotlib, seaborn
- **Interpretability** — SHAP
- **Serialization** — joblib

---

## 📝 License

This project is for educational and portfolio purposes.

---

*Built by Abdalaziz Saif*
