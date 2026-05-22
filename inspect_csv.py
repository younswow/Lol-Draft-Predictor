import pandas as pd


try:
    df = pd.read_csv('lol_dataset_challenger_grandmaster_clean.csv', encoding='latin-1', nrows=5)
    print("CSV LOADED")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
except Exception as e:
    print(f"Error: {e}")
