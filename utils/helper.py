import pandas as pd
import os

def load_csv(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"File {file_path} tidak ditemukan.")

def save_csv(df, file_path):
    df.to_csv(file_path, index=False)
    print(f"Data disimpan ke {file_path}")
