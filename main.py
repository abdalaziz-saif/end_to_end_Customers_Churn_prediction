
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score , recall_score , precision_score , f1_score , roc_auc_score , confusion_matrix , classification_report

from src.EDA import *
from src.features_engineering import *
from src.modeling import *
from src.utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Telco Customer Churn Pipline")
    parser.add_argument("--data_dir",   default="data/",    help="Directory with CSV files")
    parser.add_argument("--models_dir", default="models/",  help="Directory to save models")
    parser.add_argument("--output_dir", default="outputs/", help="Directory for submission")
    parser.add_argument("--skip_eda",   action="store_true", help="Skip EDA plots")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(f"{args.data_dir}/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print ("Dataset shape: ", df.shape)

    df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors= 'coerce')
    df = df.fillna(df['TotalCharges'].mean())
    # drop customerId columns
    df = df.drop('customerID' , axis = 1 )

    df['SeniorCitizen'] = df['SeniorCitizen'].map({1 :'yes' , 0 : 'no'})

    # Numeric columns
    numeric_columns = df.select_dtypes(include=[int, float]).columns.tolist()
    non_numeric_col =  [col for col in df.columns if col not in numeric_columns]
    
    # ____________________________________
    # EDA 
    #_______________________________________
    if not args.skip_eda:
        categorical_bivariate(df , non_numeric_col, args.output_dir)
        numerical_univariate(df , numeric_columns, args.output_dir)
        box_plot_bivariate(df , numeric_columns, args.output_dir)
        violin_plot_bivariate(df , args.output_dir)
        correlation_matrix(df , args.output_dir)
        pairplot(df , args.output_dir)
    

    #____________________________________________
    # Features Engineering 
    #____________________________________________

    # add new featues 
    df = add_features(df)
    # encode categorical features 
    df = encode_categorical(df)

    #____________________________________________
    # Features Selection 
    #____________________________________________


    Y = df['Churn']
    X = df.drop('Churn' , axis =1)

    X_train , X_ , y_train , y_ = train_test_split(X ,Y , test_size=0.4 ,random_state=42 , stratify=Y)

    X_val , X_test , y_val , y_test = train_test_split(X_ ,y_ , test_size=0.2 ,random_state=42 , stratify=y_)
    
    #______________________________________________
    # Modeling 
    # _____________________________________________
    

    # Cross Validation Function
    kf = make_kfold()

    
    results , models = train_all_models(X_train, y_train, X_val, y_val, kf)

    # Plot all results  
    print (results)
    # plot Function 
    plot_results(results, args.output_dir)

    #______________________________________________
    # Save Models
    # _____________________________________________

    # Build artifacts dict: each model + the results table
    artifacts = {f"{name}.pkl": model for name, model in models.items()}
    artifacts["results.pkl"] = results

    save_models(artifacts, models_dir=args.models_dir)
    print(f"\nAll models saved to '{args.models_dir}'")


if __name__ == '__main__' :

    main()