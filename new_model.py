import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib
import os

# Cargar datos de entrenamiento
X_train = pd.read_csv('data/processed/X_train.csv', index_col=0)
y_train = pd.read_csv('data/processed/y_train.csv', index_col=0).squeeze()

# Cargar datos de prueba (opcional para evaluación)
X_test = pd.read_csv('data/processed/X_test.csv', index_col=0)
y_test = pd.read_csv('data/processed/y_test.csv', index_col=0).squeeze()

print("Datos cargados exitosamente")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Crear y entrenar el modelo Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nEntrenando modelo Random Forest...")
rf_model.fit(X_train, y_train)
print("Modelo entrenado exitosamente")

# Evaluar en datos de entrenamiento
y_pred_train = rf_model.predict(X_train)
y_pred_proba_train = rf_model.predict_proba(X_train)[:, 1]

print("\n=== Evaluación en datos de ENTRENAMIENTO ===")
print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_train, y_pred_proba_train):.4f}")
print("\nReporte de clasificación:")
print(classification_report(y_train, y_pred_train))

# Evaluar en datos de prueba
y_pred_test = rf_model.predict(X_test)
y_pred_proba_test = rf_model.predict_proba(X_test)[:, 1]

print("\n=== Evaluación en datos de PRUEBA ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_test):.4f}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_test))

# Importancia de características
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Top 10 Características Más Importantes ===")
print(feature_importance.head(10))

# Guardar el modelo
os.makedirs('models', exist_ok=True)
joblib.dump(rf_model, 'models/random_forest_model.pkl')
print("\nModelo guardado en 'models/random_forest_model.pkl'")