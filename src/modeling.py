"""
Modeling Module

    - Cross-validation
    - Evaluation function
    - Model training (Logistic Regression, KNN, Random Forest, XGBoost, CatBoost, GBM)
    - Results comparison
    - Feature importance (SHAP)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

# _____________________________________________

#  Cross-validation Function
# ─────────────────────────────────────────────

def make_kfold(n_splits: int = 5, random_state: int = 42) -> KFold:
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


#_______________________________________________

# Evaluateion Function
#_______________________________________________


def evaluation_function(name: str, model,
                        X_train: pd.DataFrame, y_train: pd.DataFrame,
                        X_val: pd.DataFrame, y_val: pd.DataFrame, kf: KFold):

    """
    Evaluate The model
        - Apply CrossValidation To See how the Model will Generlize (how it work with unseen data)
        - Calc The accuracy
        - Check The overfiting BY comparing the Train accuracy and Validation
    """
    cv = []
    for train_idx, val_idx in kf.split(X_train):

        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_vl = X_train.iloc[val_idx]
        y_vl = y_train.iloc[val_idx]

        model1 = clone(model)
        model1.fit(X_tr, y_tr)
        pred = model1.predict(X_vl)
        acc = accuracy_score(y_true=y_vl, y_pred=pred)
        cv.append(acc)

    # calc the average cross validation score
    cv_acc = np.mean(cv)

    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc   = accuracy_score(y_val,   model.predict(X_val))


    print(f"{name}:")
    print(f"  CV Acc    : {cv_acc:.2f}% \u00b1 {np.std(cv):.4f}")
    print(f"  Train Acc : {train_acc:.2f}%")
    print(f"  Val Acc   : {val_acc:.2f}%")

    return cv_acc, val_acc, model


#_______________________________________________

# Model Training Functions
#_______________________________________________


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.DataFrame,
                              X_val: pd.DataFrame, y_val: pd.DataFrame,
                              kf: KFold):
    """
    Train Logistic Regression with GridSearchCV
    """
    lr_params = {
        "C": np.logspace(-4, 4, 20),
        "solver": ["liblinear"]
    }

    lr_grid = GridSearchCV(
        LogisticRegression(),
        lr_params,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    lr_grid.fit(X_train, y_train)

    best_lr = lr_grid.best_estimator_

    print(f"Best LR params: {lr_grid.best_params_}")

    lr_cv, lr_vl, lr_model = evaluation_function(
        "Logistic_Regression", best_lr, X_train, y_train, X_val, y_val, kf
    )

    return lr_cv, lr_vl, lr_model


def train_knn(X_train: pd.DataFrame, y_train: pd.DataFrame,
              X_val: pd.DataFrame, y_val: pd.DataFrame,
              kf: KFold):
    """
    Train KNN Classifier with GridSearchCV
    """
    knn_params = {
        'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    }

    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        knn_params,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    knn_grid.fit(X_train, y_train)

    best_knn = knn_grid.best_estimator_

    print(f"KNN Best Params: {knn_grid.best_params_}")

    knn_cv, knn_vl, knn_model = evaluation_function(
        "KNN Classifier", best_knn, X_train, y_train, X_val, y_val, kf
    )

    return knn_cv, knn_vl, knn_model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.DataFrame,
                        X_val: pd.DataFrame, y_val: pd.DataFrame,
                        kf: KFold):
    """
    Train Random Forest with GridSearchCV
    """
    rf_params = {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10]
    }

    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        rf_params,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    rf_grid.fit(X_train, y_train)

    rf_best = rf_grid.best_estimator_

    print(f"Best RF params: {rf_grid.best_params_}")

    rf_cv, rf_vl, rf_model = evaluation_function(
        "Random_Forest", rf_best, X_train, y_train, X_val, y_val, kf
    )

    return rf_cv, rf_vl, rf_model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.DataFrame,
                  X_val: pd.DataFrame, y_val: pd.DataFrame,
                  kf: KFold):
    """
    Train XGBoost with RandomizedSearchCV
    """
    xgb_params = dict(
        n_estimators=stats.randint(10, 1000),
        max_depth=stats.randint(1, 10),
        learning_rate=stats.uniform(0, 1)
    )

    xgb_clf = XGBClassifier()

    xgb_rand = RandomizedSearchCV(
        xgb_clf, xgb_params, cv=5, n_iter=150,
        scoring='accuracy', n_jobs=-1, verbose=1
    )

    xgb_rand.fit(X_train, y_train)

    xgb_best = xgb_rand.best_estimator_

    print(f"Best XGB params: {xgb_rand.best_params_}")

    xgb_cv, xgb_vl, xgb_model = evaluation_function(
        "XGBOOST", xgb_best, X_train, y_train, X_val, y_val, kf
    )

    return xgb_cv, xgb_vl, xgb_model


def train_catboost(X_train: pd.DataFrame, y_train: pd.DataFrame,
                   X_val: pd.DataFrame, y_val: pd.DataFrame,
                   kf: KFold):
    """
    Train CatBoost Classifier
    """
    best_cat = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        eval_metric='Accuracy',
        verbose=False
    )

    best_cat.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50
    )

    cat_cv, cat_val, best_cat = evaluation_function(
        'CatBoost', best_cat, X_train, y_train, X_val, y_val, kf
    )

    return cat_cv, cat_val, best_cat


def train_gbm(X_train: pd.DataFrame, y_train: pd.DataFrame,
              X_val: pd.DataFrame, y_val: pd.DataFrame,
              kf: KFold):
    """
    Train Gradient Boosting Machine with GridSearchCV
    """
    gbm_params = {
        "n_estimators": [100, 300],
        "learning_rate": [0.001, 0.01, 0.05, 0.1],
        "max_depth": [2, 3]
    }

    gbm_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        gbm_params,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    gbm_grid.fit(X_train, y_train)

    best_gbm = gbm_grid.best_estimator_

    print(f"Best GBM params: {gbm_grid.best_params_}")

    gbm_cv, gbm_val, best_gbm = evaluation_function(
        "GBM", best_gbm, X_train, y_train, X_val, y_val, kf
    )

    return gbm_cv, gbm_val, best_gbm


#_______________________________________________

# Train All Models
#_______________________________________________

def train_all_models(X_train: pd.DataFrame, y_train: pd.DataFrame,
                     X_val: pd.DataFrame, y_val: pd.DataFrame,
                     kf: KFold) -> pd.DataFrame:
    """
    Train all models and return a results DataFrame
    """
    lr_cv, lr_vl, lr_model = train_logistic_regression(X_train, y_train, X_val, y_val, kf)
    knn_cv, knn_vl, knn_model = train_knn(X_train, y_train, X_val, y_val, kf)
    rf_cv, rf_vl, rf_model = train_random_forest(X_train, y_train, X_val, y_val, kf)
    xgb_cv, xgb_vl, xgb_model = train_xgboost(X_train, y_train, X_val, y_val, kf)
    cat_cv, cat_vl, cat_model = train_catboost(X_train, y_train, X_val, y_val, kf)
    gbm_cv, gbm_vl, gbm_model = train_gbm(X_train, y_train, X_val, y_val, kf)

    results = pd.DataFrame(
        {
            'models': ['Logistic Regression', 'KNN Classifier', 'Random Forest',
                        'Xgboost', 'Catboost', 'Gbm'],
            'accuracy': [lr_vl, knn_vl, rf_vl, xgb_vl, cat_vl, gbm_vl],
            'cv_acc':   [lr_cv, knn_cv, rf_cv, xgb_cv, cat_cv, gbm_cv],
        }
    )

    models = {
        'Logistic Regression': lr_model,
        'KNN Classifier': knn_model,
        'Random Forest': rf_model,
        'Xgboost': xgb_model,
        'Catboost': cat_model,
        'Gbm': gbm_model,
    }

    return results, models


#_______________________________________________

# Results Visualization
#_______________________________________________

def plot_results(results: pd.DataFrame, outputdir: str = None) -> None:
    """
    Plot validation accuracy bar chart and CV vs Val accuracy line chart
    """


    # Bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results, x='models', y='accuracy', palette='Set2')
    plt.title('Val Accuracy For All Models')
    plt.xlabel('Models')
    plt.ylabel('Validation Accuracy')
    plt.xticks(rotation=15)
    if outputdir:
        os.makedirs(outputdir, exist_ok=True)
        plt.savefig(os.path.join(outputdir, 'model_val_accuracy.png'))
    plt.show()

    # Line plot
    plt.figure(figsize=(10, 6))
    plt.plot(results['models'], results['accuracy'], label='Val accuracy')
    plt.plot(results['models'], results['cv_acc'], label='CV accuracy')
    plt.title("Models Accuracy")
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    if outputdir:
        os.makedirs(outputdir, exist_ok=True)
        plt.savefig(os.path.join(outputdir, 'model_accuracy_comparison.png'))
    plt.show()


#_______________________________________________

# Feature Importance (SHAP)
#_______________________________________________

def shap_feature_importance(model, X_val: pd.DataFrame , outputdir: str = None) -> None:
    """
    Compute and plot SHAP feature importance for a given model
    """

    explainer = shap.Explainer(model)
    shap_values = explainer(X_val)
    shap.summary_plot(shap_values)
    plt.show()
    shap.plots.bar(shap_values)
    plt.show()
    if outputdir:
        os.makedirs(outputdir, exist_ok=True)
        plt.savefig(os.path.join(outputdir, 'shap_summary_plot.png'))
        plt.savefig(os.path.join(outputdir, 'shap_bar_plot.png'))

