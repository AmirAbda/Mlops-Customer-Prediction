# File: src/data/data_loader.py

import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Tuple

def load_config(config_path: str = "configs/model_config.yaml") -> Dict:
    """
    Load the configuration from the YAML file.
    
    Args:
        config_path (str): Path to the configuration file.
    
    Returns:
        Dict: Configuration dictionary.
    
    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    try:
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing the YAML configuration file: {e}")

def load_data(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test data from CSV files specified in the configuration.
    
    Args:
        config (Dict): Configuration dictionary containing file paths.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test DataFrames.
    
    Raises:
        FileNotFoundError: If either train or test CSV file is not found.
        pd.errors.EmptyDataError: If either CSV file is empty.
        Exception: For any other unexpected errors during data loading.
    """
    try:
        train_path = Path(config['data']['raw']['train'])
        test_path = Path(config['data']['raw']['test'])
        
        if not train_path.exists():
            raise FileNotFoundError(f"Train file not found at {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found at {test_path}")
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        if train_df.empty:
            raise pd.errors.EmptyDataError("Train CSV file is empty")
        if test_df.empty:
            raise pd.errors.EmptyDataError("Test CSV file is empty")
        
        return train_df, test_df
    
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(f"Error: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading the data: {e}")

if __name__ == "__main__":
    try:
        config = load_config()
        train_data, test_data = load_data(config)
        print("Data loaded successfully.")
        print(f"Train data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
    except Exception as e:
        print(f"Error: {e}")