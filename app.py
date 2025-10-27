import streamlit as st
import pandas as pd
from pathlib import Path
from src.inference_pipeline import predict_churn  # función que implementaste en inference_pipeline.py

base = Path(__file__).resolve().parent.parent  # carpeta raíz del proyecto

THRESHOLD = 0.25  # umbral ajustado para decisión final

def main():
    st.set_page_config(
        page_title="Predictor de Churn",
        page_icon="📊",
        layout="wide"
    )

    st.title('Predictor de Churn para Clientes de Telecom')
    st.write("""
    Esta aplicación predice la probabilidad de que un cliente abandone el servicio (churn).
    Complete los siguientes campos con la información del cliente para obtener una predicción.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Información Demográfica")
        gender = st.selectbox('Género', ['Female', 'Male'])
        senior_citizen = st.selectbox('Cliente Senior', ['No', 'Yes'])
        partner = st.selectbox('¿Tiene pareja?', ['No', 'Yes'])
        dependents = st.selectbox('¿Tiene dependientes?', ['No', 'Yes'])

    with col2:
        st.subheader("Servicios Contratados")
        phone_service = st.selectbox('Servicio Telefónico', ['No', 'Yes'])
        multiple_lines = st.selectbox('Múltiples Líneas', ['No', 'Yes', 'No phone service'])
        internet_service = st.selectbox('Servicio de Internet', ['DSL', 'Fiber optic', 'No'])
        online_security = st.selectbox('Seguridad en Línea', ['No', 'Yes', 'No internet service'])
        online_backup = st.selectbox('Respaldo en Línea', ['No', 'Yes', 'No internet service'])
        device_protection = st.selectbox('Protección de Dispositivo', ['No', 'Yes', 'No internet service'])
        tech_support = st.selectbox('Soporte Técnico', ['No', 'Yes', 'No internet service'])
        streaming_tv = st.selectbox('TV Streaming', ['No', 'Yes', 'No internet service'])
        streaming_movies = st.selectbox('Películas Streaming', ['No', 'Yes', 'No internet service'])

    with col3:
        st.subheader("Información de Contrato")
        contract = st.selectbox('Tipo de Contrato', ['Month-to-month', 'One year', 'Two year'])
        paperless_billing = st.selectbox('Facturación sin Papel', ['No', 'Yes'])
        payment_method = st.selectbox('Método de Pago', 
                                      ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        tenure = st.slider('Tiempo como Cliente (meses)', 0, 72, 1)
        monthly_charges = st.number_input('Cargo Mensual ($)', min_value=0.0, max_value=1000.0, value=50.0)
        total_charges = st.number_input('Cargos Totales ($)', min_value=0.0, max_value=10000.0, value=0.0)

    if st.button('Predecir Churn'):
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })

        try:
            churn_prediction, proba = predict_churn(input_data)
            prediction = "Sí" if proba >= THRESHOLD else "No"

            st.subheader('Resultados de la Predicción')
            col_a, col_b = st.columns(2)

            with col_a:
                st.metric("Probabilidad de Churn", f"{proba:.1%}")
            with col_b:
                st.metric("¿Cliente en Riesgo de Abandono?", prediction)

            if prediction == 'Sí':
                st.warning("""
                ⚠️ ALERTA: Este cliente tiene un alto riesgo de abandono.
                Se recomienda tomar acciones proactivas para retener al cliente.
                """)

                st.subheader("Recomendaciones")
                if contract == 'Month-to-month':
                    st.write("- Ofrecer un contrato a largo plazo con descuento")
                if internet_service == 'Fiber optic' and (online_security == 'No' or tech_support == 'No'):
                    st.write("- Sugerir paquete de seguridad y soporte técnico")
                if monthly_charges > 100:
                    st.write("- Revisar posibilidad de ajuste en el plan actual")
            else:
                st.success("""
                ✅ BAJO RIESGO: Este cliente muestra un bajo riesgo de abandono.
                Se recomienda mantener el nivel de servicio actual.
                """)
        except Exception as e:
            st.error(f"Error en la predicción: {e}")

    st.markdown("""
    ---
    ### Información sobre el Modelo

    Este predictor utiliza un modelo XGBoost entrenado con datos históricos de clientes.
    El umbral de decisión está establecido en 0.25 para maximizar la detección de posibles casos de abandono.
    """)

if __name__ == "__main__":
    main()