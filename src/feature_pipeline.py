import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def clean_data(df):
    '''Limpia el DataFrame de entrada.'''
    df = df.copy()
    df = df.drop(columns=["customerID"], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df


def get_column_types(df):
    '''Devuelve listas de columnas numéricas y categóricas.'''
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    categorical_cols = [col for col in df.columns if col not in numerical_cols]
    return numerical_cols, categorical_cols


def build_feature_pipeline(num_cols, cat_cols):
    '''Construye y devuelve un pipeline de preprocesamiento de features.'''
    num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
    ])


    cat_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])


    full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
    ])


    return full_pipeline