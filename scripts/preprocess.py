import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def preprocess_data(df, fit_scaler=True, fit_encoder=True):
    """
    Preprocesses the input dataset:
    - Encodes the target variable
    - Standardizes numerical features
    
    Parameters:
    df (pd.DataFrame): Input data containing soil parameters and crop labels
    fit_scaler (bool): Whether to fit a new StandardScaler (True for training, False for inference)
    fit_encoder (bool): Whether to fit a new LabelEncoder (True for training, False for inference)
    
    Returns:
    tuple: (Preprocessed feature matrix X, Encoded target variable y, scaler, label_encoder)
    """
    # Separate features and target
    X = df.drop(columns=['crop'])  # Ensure 'crop' is the target variable
    y = df['crop']
    
    # Encode the target variable
    label_encoder = LabelEncoder()
    if fit_encoder:
        y_encoded = label_encoder.fit_transform(y)
        joblib.dump(label_encoder, 'models/label_encoder.pkl')  # Save encoder
    else:
        label_encoder = joblib.load('models/label_encoder.pkl')
        y_encoded = label_encoder.transform(y)
    
    # Standardize numerical features
    numerical_features = ['N', 'P', 'K', 'ph']
    X[numerical_features] = X[numerical_features].astype('float64')
    scaler = StandardScaler()
    if fit_scaler:
        X_scaled = X.copy()
        X_scaled[numerical_features] = scaler.fit_transform(X[numerical_features])
        joblib.dump(scaler, 'models/scaler.pkl')  # Save scaler
    else:
        scaler = joblib.load('models/scaler.pkl')
        X_scaled = X.copy()
        X_scaled[numerical_features] = scaler.transform(X[numerical_features])
    
    return X_scaled, y_encoded, scaler, label_encoder

def preprocess_input(input_df):
    """
    Prepares new input data for model prediction.
    
    Parameters:
    input_df (pd.DataFrame): User input containing soil parameters
    
    Returns:
    pd.DataFrame: Scaled input ready for prediction
    """
    scaler = joblib.load('models/scaler.pkl')  # Load pre-trained scaler
    numerical_features = ['N', 'P', 'K', 'ph']
    input_df[numerical_features] = input_df[numerical_features].astype('float64')
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])
    
    return input_df