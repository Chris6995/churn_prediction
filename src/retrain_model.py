# reentrenamiento.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.feature_pipeline import clean_data, build_feature_pipeline, get_column_types
from src.training_pipeline import train_model

import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE

# 1. Cargar y limpiar datos
df_raw = pd.read_csv("data/raw/customer_churn.csv")
df_cleaned = clean_data(df_raw)

X = df_cleaned.drop("Churn", axis=1)
y = df_cleaned["Churn"].map({"Yes": 1, "No": 0})

# 2. Sacar columnas numéricas y categóricas y construir pipeline
numerical_cols, categorical_cols = get_column_types(X)

# 3. Construir y ajustar el pipeline de features
pipeline = build_feature_pipeline(numerical_cols, categorical_cols)
X_processed = pipeline.fit_transform(X)

# 4. Guardar el feature pipeline
joblib.dump(pipeline, "models/feature_pipeline.pkl")


# 5. Balancear clases con SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_processed, y)


# 6. Entrenar modelo y guardarlo
model = train_model(X_train_bal, y_train_bal)

print("✅ Reentrenamiento completo.")
