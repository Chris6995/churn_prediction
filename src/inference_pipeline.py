import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from src.feature_pipeline import clean_data

# Cargar los artefactos una vez
pipeline = joblib.load("models/feature_pipeline.pkl")
model = joblib.load("models/xgb_model.pkl")
try:
    threshold = joblib.load("models/best_threshold.pkl")
except:
    threshold = 0.25  # Valor por defecto
        

def predict_churn(df_raw: pd.DataFrame) -> tuple[str, float]:
    """
    Toma un DataFrame con los datos crudos (una fila), lo limpia,
    aplica el feature pipeline, predice con el modelo y devuelve
    predicción (Sí/No) y probabilidad.
    """
    # 1. Limpiar datos
    df_clean = clean_data(df_raw)

    # 2. Aplicar pipeline
    X = pipeline.transform(df_clean)

    # 3. Predecir
    probas = model.predict_proba(X)[:, 1]
    preds = np.where(probas >= threshold, "Sí", "No")

    return preds[0], probas[0]
