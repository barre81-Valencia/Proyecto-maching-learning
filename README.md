🍷 Clasificación de Calidad de Vino (Blanco y Tinto) - Proyecto de Machine Learning

🎯 Objetivo del Proyecto

El objetivo principal de este proyecto es desarrollar y optimizar un modelo de Machine Learning capaz de predecir la calidad de los vinos (blanco y tinto) basándose en 11 atributos fisicoquímicos.

El resultado final es una aplicación interactiva (Streamlit) que permite a los usuarios ingresar los parámetros de un vino hipotético para obtener una clasificación inmediata: Malo (3-5), Regular (6) o Bueno (7-9).

💻 Arquitectura del Sistema

El proyecto se divide en tres componentes principales:

Exploración y Preprocesamiento de Datos: Limpieza, normalización y Feature Engineering (creación de la variable type_white).

Entrenamiento y Optimización de Modelos: Comparativa de modelos de Regresión y Clasificación.

Aplicación Web (Streamlit): Interfaz de usuario para la predicción en tiempo real.

🏆 Modelo Seleccionado: AdaBoost Regressor Optimizado

Tras una evaluación exhaustiva de modelos (incluyendo Random Forest y XGBoost), se seleccionó una versión optimizada del AdaBoost Regressor por su equilibrio superior entre precisión general y capacidad de predecir correctamente la clase minoritaria (Malo).

Métricas Clave

Métrica

AdaBoost Optimizado

Random Forest Optimizado

Accuracy General

0.73

0.70

F1-Score Ponderado

0.73

0.69

F1-Score Clase Malo (0)

0.77

0.75

El AdaBoost Regressor demostró ser más robusto y efectivo para manejar la naturaleza continua de la variable de calidad (score 3-9) y las tres clases discretas resultantes.

Mapeo de Clases

Para la predicción final, se utilizaron los siguientes umbrales en el output continuo del modelo:

Clase

Predicción Continua

Score Original (Calidad)

Malo

< 0.75

3, 4, 5

Regular

0.75 - 0.94

6

Bueno

>= 0.95

7, 8, 9

📊 Interpretación de Features (Correlación)

El análisis de correlación lineal indica qué features tienen mayor impacto (positivo o negativo) en la calidad final del vino:

Feature

Coeficiente

Impacto

Alcohol

Positiva Fuerte (+0.45)

Es el factor más influyente. A mayor alcohol, mayor calidad.

Densidad

Negativa Fuerte (-0.32)

A mayor densidad, menor calidad.

Acidez Volátil

Negativa Media (-0.22)

Es un gran penalizador de la calidad (indica deterioro).

Sulfatos

Positiva Débil (+0.07)

Tienen una ligera correlación positiva.

🚀 Cómo Ejecutar la Aplicación

Requisitos

Asegúrate de tener Python 3.8+ instalado y las siguientes librerías:

pip install pandas scikit-learn streamlit joblib


Archivos Necesarios

Para que la aplicación funcione, deben estar presentes los siguientes archivos en el mismo directorio que app.py:

app.py (La aplicación Streamlit).

modelo_final_adaboost_campeon.pkl (El modelo entrenado).

scaler_fit_campeon.pkl (El escalador ajustado para normalizar los datos de entrada).

Ejecución

Navega hasta el directorio del proyecto en tu terminal y ejecuta el siguiente comando:

streamlit run app.py


La aplicación se abrirá automáticamente en tu navegador web.

Desarrollado por: David Barrero V
Proyecto semana 7 - Boot Camp Análisis de Datos / Ironhack
Fecha de finalización: Noviembre 21 de 2025
