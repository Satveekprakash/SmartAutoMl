import pandas as pd

def load_data(path):
    path=str(path)
    try:
        if path.endswith(".csv"):
            return pd.read_csv(path)
        elif path.endswith(".xlsx"):
            return pd.read_excel(path)
    except Exception as e:
        print("Error loading file:", e)
        return None