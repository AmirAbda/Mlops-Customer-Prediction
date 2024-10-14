# File: src/data/data_preprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Dict, Tuple
from pathlib import Path

def preprocess_data(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Preprocess the input DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame to preprocess.
        config (Dict): Configuration dictionary.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = df.copy()  # Create a copy to avoid modifying the original DataFrame
    
    # Remove CustomerID column
    df = df.drop(columns='CustomerID', errors='ignore')
    
    # Handle missing values
    df = df.dropna()
    
    # Encode categorical variables
    categorical_features = ['Gender', 'Subscription Type', 'Contract Length']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    
    # Drop original categorical columns and concatenate encoded features
    df = df.drop(columns=categorical_features)
    df = pd.concat([df, encoded_df], axis=1)
    
    # Scale numerical features
    numerical_features = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    return df

def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the DataFrame into features and target.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target.
    """
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    return X, y

def save_processed_data(df: pd.DataFrame, config: Dict) -> None:
    """
    Save the processed DataFrame to a CSV file.
    
    Args:
        df (pd.DataFrame): Processed DataFrame to save.
        config (Dict): Configuration dictionary.
    """
    try:
        output_path = Path(config['data']['processed']['output'])
    except KeyError:
        # If 'processed' key is not in config, use a default path
        output_path = Path("data/processed/processed_data.csv")
        print(f"Warning: 'processed' key not found in config. Using default path: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    from data_loader import load_config, load_data
    
    try:
        config = load_config()
        train_data, test_data = load_data(config)
        
        # Preprocess train and test data
        processed_train = preprocess_data(train_data, config)
        processed_test = preprocess_data(test_data, config)
        
        # Combine processed train and test data
        processed_data = pd.concat([processed_train, processed_test], axis=0)
        
        # Save processed data
        save_processed_data(processed_data, config)
        
        print("Data preprocessing completed successfully.")
        print(f"Processed data shape: {processed_data.shape}")
        
    except Exception as e:
        print(f"Error: {e}")