"""
Feature Engineering Module 



"""
import pandas as pd 
import numpy as np 
import os 
from sklearn.preprocessing import LabelEncoder


#________________________________________________________________________________

# Add New Features  
#_________________________________________________________________________________

# add Number of services feature
num_services = service_cols = ['OnlineSecurity_Yes', 'OnlineBackup_Yes',
      'DeviceProtection_Yes', 'TechSupport_Yes',
      'StreamingTV_Yes', 'StreamingMovies_Yes']

def add_features(df: pd.DataFrame) -> pd.DataFrame:

    df['Num_of_services'] = df[[c for c in num_services if c in df.columns]].sum(axis=1)

    # add tenure bucket
    df['tenure_bucket'] = pd.cut(df['tenure'],
                          bins=[0, 6, 12, 24, 36, 48, 60, 72],
                          labels=['0-6m', '7-12m', '13-24m', '25-36m', '37-48m', '49-60m', '60m+'])

    return df


#________________________________________________________________________________

# Encode Categorical Features
#_________________________________________________________________________________

# binaray features columns 
binary_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:

  # binary Encoding
  le = LabelEncoder()
  for col in binary_features:
    df[col] = le.fit_transform(df[col])

  # OneHot Encoding

  onehot_features = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod', 'tenure_bucket']

  df = pd.get_dummies(df, columns=onehot_features, drop_first=True)

  return df
