# Proyecto de Predicción de Churn

Este proyecto implementa un modelo de machine learning para predecir el churn (abandono) de clientes, junto con una interfaz web construida con Streamlit para su despliegue.

## Estructura del Proyecto

```
├── data/
│   ├── raw/          # Datos sin procesar
│   └── processed/    # Datos procesados
├── models/           # Modelos entrenados
├── notebooks/        # Jupyter notebooks para análisis y desarrollo
├── src/             # Código fuente de la aplicación
├── requirements.txt  # Dependencias del proyecto
└── README.md        # Este archivo
```

## Configuración del Entorno

1. Crear entorno virtual:
```bash
python -m venv venv
```

2. Activar el entorno virtual:
```bash
source venv/bin/activate  # En macOS/Linux
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

1. El análisis exploratorio y desarrollo del modelo se encuentra en los notebooks.
2. Los modelos entrenados se guardan en la carpeta `models/`.
3. La aplicación Streamlit se encuentra en `src/`.

## Autor
[Tu nombre]

## Licencia
[Tipo de licencia]