import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessor(df):
    categorical_cols = df.select_dtypes(include="object").columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ])

    return preprocessor
