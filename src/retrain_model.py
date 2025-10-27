# reentrenamiento.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.feature_pipeline import clean_data, build_feature_pipeline, get_column_types
from src.training_pipeline import train_model

import pandas as pd
import joblib

# 1. Cargar y limpiar datos
df_raw = pd.read_csv("data/raw/customer_churn.csv")
df_cleaned = clean_data(df_raw)

X = df_cleaned.drop("Churn", axis=1)
y = df_cleaned["Churn"].map({"Yes": 1, "No": 0})

# 2. Crear y ajustar feature pipeline
numerical_cols, categorical_cols = get_column_types(X)


pipeline = joblib.load("models/feature_pipeline.pkl")

print("Numéricas:", pipeline.transformers_[0][2])
print("Categóricas:", pipeline.transformers_[1][2])

print(X.columns)

pipeline = build_feature_pipeline(numerical_cols, categorical_cols)
X_processed = pipeline.fit_transform(X)

# 3. Guardar el feature pipeline
joblib.dump(pipeline, "models/feature_pipeline.pkl")

# 4. Entrenar modelo
model = train_model(X_processed, y)

print("✅ Reentrenamiento completo.")
