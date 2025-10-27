import streamlit as st
import pandas as pd
import joblib
import numpy as np

def load_model():
    """Carga el modelo entrenado"""
    try:
        model = joblib.load('../models/best_model.joblib')
        return model
    except:
        st.error("No se pudo cargar el modelo. Asegúrate de que existe en la carpeta models/")
        return None

def main():
    st.title("Predictor de Churn")
    st.write("""
    ### Predicción de abandono de clientes
    Esta aplicación predice la probabilidad de que un cliente abandone el servicio.
    """)

    # Aquí irán los inputs para las características del cliente
    st.sidebar.header("Características del Cliente")
    
    # Ejemplo de inputs (ajustar según el modelo final)
    tenure = st.sidebar.slider("Tiempo como cliente (meses)", 0, 72, 36)
    monthly_charges = st.sidebar.number_input("Cargo mensual ($)", 0.0, 200.0, 70.0)
    total_charges = st.sidebar.number_input("Cargo total ($)", 0.0, 8000.0, 2000.0)
    
    # Botón para realizar predicción
    if st.sidebar.button("Predecir Churn"):
        model = load_model()
        if model:
            # Preparar datos para la predicción
            features = pd.DataFrame({
                'tenure': [tenure],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges]
            })
            
            # Realizar predicción
            prediction = model.predict_proba(features)
            
            # Mostrar resultado
            st.subheader("Resultados de la Predicción")
            churn_prob = prediction[0][1]
            st.write(f"Probabilidad de abandono: {churn_prob:.2%}")
            
            # Visualización del resultado
            if churn_prob < 0.3:
                st.success("Cliente con bajo riesgo de abandono")
            elif churn_prob < 0.6:
                st.warning("Cliente con riesgo moderado de abandono")
            else:
                st.error("Cliente con alto riesgo de abandono")

if __name__ == "__main__":
    main()