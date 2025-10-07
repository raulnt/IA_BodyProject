import pandas as pd
import numpy as np

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    df = df.dropna()
    
    return df
