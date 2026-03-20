import pandas as pd
def detect_problem_type(data, target):

    if pd.api.types.is_numeric_dtype(data[target]):
        return "Regression"
    else:
        return "Classification"