import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Calidad de Vino",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded" 
)

st.title("🍷 Clasificador de Calidad de Vino: Proyecto Final")
st.markdown("---")

st.subheader("🛠️ Parámetros Físico-Químicos del Vino")
st.markdown("Introduce los 11 *features* para que el modelo AdaBoost Optimizado V2 prediga la calidad del vino (Malo, Regular, Bueno).")

# -------------------------------------------------------------
# SIDEBAR (Contexto del Proyecto)
# -------------------------------------------------------------

st.sidebar.title("📚 Contexto del Proyecto")
st.sidebar.markdown("Este clasificador fue desarrollado como parte de un proyecto de Machine Learning, con el objetivo de predecir la calidad del vino (blanco y tinto) basándose en su composición fisicoquímica.")

st.sidebar.subheader("Modelo Campeón")
st.sidebar.info("**AdaBoost Optimizado**")
st.sidebar.markdown(
    """
    **Target Final:** 3 Clases V2 (Malo, Regular, Bueno)
    **Métricas Clave:**
    * **Accuracy:** 73%
    * **F1-Score Ponderado:** 0.73
    """
)

st.sidebar.subheader("Definición de Clases")
st.sidebar.markdown(
    """
    * **Bueno:** Calidad (Score) 7, 8, 9
    * **Regular:** Calidad (Score) 6
    * **Malo:** Calidad (Score) 3, 4, 5
    """
)

# 1. Cargar el Modelo Campeón y el SCALER
# Cargar el modelo
try:
    model = joblib.load('modelo_final_adaboost_campeon.pkl')
except FileNotFoundError:
    st.error("Error: Archivo de modelo 'modelo_final_adaboost_campeon.pkl' no encontrado.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()
    
# Cargar el SCALER (CRÍTICO: Necesario para normalizar los datos del usuario)
try:
    # Asegúrate de que este archivo contenga el StandardScaler o MinMaxScaler ajustado
    scaler = joblib.load('scaler_fit_campeon.pkl')
except FileNotFoundError:
    st.error("Error: Archivo de escalado 'scaler_fit_campeon.pkl' no encontrado. Asegúrate de guardarlo en tu notebook.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar el escalador (scaler): {e}")
    st.stop()


# 2. Mapeo de la Predicción a Etiquetas Comprensibles
CLASE_MAPEO = {
    0: "Mala (Puntaje <= 5)",    
    1: "Regular (Puntaje = 6)",  
    2: "Buena (Puntaje >= 7)"       
}

# Lista de nombres de columna para mantener el orden estricto (CRÍTICO)
FEATURE_NAMES = [
    'fixed acidity', 
    'volatile acidity', 
    'citric acid', 
    'residual sugar',
    'chlorides', 
    'free sulfur dioxide', 
    'total sulfur dioxide', 
    'density',
    'pH', 
    'sulphates', 
    'alcohol', 
    'type_white' 
]

# 3. Función de Predicción (CON AJUSTE DE UMBRAL Y CORRECCIÓN DE WARNING)
def predict_quality(data):
    
    # ----------------------------------------------------------
    # SOLUCIÓN AL USERWARNING: Usar los nombres de columna al predecir
    # ----------------------------------------------------------
    # 1. Escalar los datos de entrada (resulta en un array de NumPy sin nombres)
    data_scaled_array = scaler.transform(data)
    
    # 2. Convertir el array escalado de vuelta a un DataFrame, 
    #    usando los nombres de columna originales de 'data'. Esto elimina el warning.
    data_scaled = pd.DataFrame(data_scaled_array, columns=data.columns)
    
    # El modelo AdaBoostRegressor predice un valor continuo
    prediction_cont = model.predict(data_scaled)[0] 
    
    # AJUSTES DE UMBRALES CONFIRMADOS:
    # Si la predicción es muy baja (< 0.75), la forzamos a 'Mala'
    if prediction_cont < 0.75:
        final_prediction = 0 # Malo
    # Si la predicción es 0.95 o superior, la forzamos a 'Buena'
    elif prediction_cont >= 0.95:
        final_prediction = 2 # Bueno
    else:
        final_prediction = 1 # Regular (entre 0.75 y 0.94)
    
    # Devolvemos la etiqueta mapeada y el valor continuo
    return CLASE_MAPEO[final_prediction], prediction_cont

# =======================================================================
# 4. INTERFAZ DE STREAMLIT (Entrada de Datos)
# =======================================================================

with st.form("input_form"):
    
    st.markdown("### 🧪 Ingreso de Parámetros")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.caption("Acidez y Cloruros")
        fixed_acidity = st.slider("Acidez Fija (g/L)", 3.0, 16.0, 7.0, 0.1)
        volatile_acidity = st.slider("Acidez Volátil (g/L)", 0.0, 2.0, 0.4, 0.01)
        citric_acid = st.slider("Ácido Cítrico (g/L)", 0.0, 2.0, 0.3, 0.01)
        chlorides = st.slider("Cloruros (g/L)", 0.01, 0.6, 0.05, 0.001)

    with col_b:
        st.caption("Azúcares y Dióxidos")
        residual_sugar = st.slider("Azúcar Residual (g/L)", 0.5, 70.0, 5.0, 0.1)
        free_sulfur_dioxide = st.slider("Dióxido de Azufre Libre (mg/L)", 1.0, 80.0, 30.0, 1.0)
        total_sulfur_dioxide = st.slider("Dióxido de Azufre Total (mg/L)", 6.0, 300.0, 100.0, 1.0)
        density = st.slider("Densidad (g/mL)", 0.98, 1.04, 0.99, 0.0001, format="%.4f")
        
    with col_c:
        st.caption("Composición Final")
        pH = st.slider("pH", 2.5, 4.5, 3.2, 0.01)
        sulphates = st.slider("Sulfatos (g/L)", 0.2, 2.0, 0.6, 0.01)
        alcohol = st.slider("Alcohol (%)", 8.0, 15.0, 10.0, 0.1)
        
        st.markdown("<br>", unsafe_allow_html=True)
        wine_type_label = st.radio("Tipo de Vino:", ('Blanco', 'Tinto'), horizontal=True)

    st.markdown("---")
    submitted = st.form_submit_button("🍷 Predecir Calidad del Vino", type="primary")

# 5. Lógica de Predicción al Enviar el Formulario
if submitted:
    type_white_value = 1 if wine_type_label == 'Blanco' else 0
    
    # Creación de DataFrame con los datos de entrada
    input_data_dict = {
        'fixed acidity': [fixed_acidity], 
        'volatile acidity': [volatile_acidity], 
        'citric acid': [citric_acid], 
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides], 
        'free sulfur dioxide': [free_sulfur_dioxide], 
        'total sulfur dioxide': [total_sulfur_dioxide], 
        'density': [density],
        'pH': [pH], 
        'sulphates': [sulphates], 
        'alcohol': [alcohol], 
        'type_white': [type_white_value] 
    }
    
    # MUY IMPORTANTE: Se crea el DataFrame forzando el orden de las columnas 
    # para coincidir con el modelo entrenado y evitar errores.
    input_data = pd.DataFrame(input_data_dict, columns=FEATURE_NAMES)
    
    # Realizar la predicción
    result, prediction_cont = predict_quality(input_data)
    
    # -------------------------------------------------------------
    # MOSTRAR RESULTADO MEJORADO
    # -------------------------------------------------------------
    
    st.markdown("<h2 style='text-align: center;'>Resultado de la Predicción</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    
    if "Mala" in result:
        st.error(f"❌ Clasificación: {result}")
        color_hex = "#dc3545" # Rojo
        estrellas = "★☆☆"
        mensaje = "Se recomienda revisar la composición. El modelo lo clasifica como **Malo**." 
    elif "Buena" in result:
        st.success(f"✅ Clasificación: {result}")
        color_hex = "#28a745" # Verde
        estrellas = "★★★"
        mensaje = "¡Excelente noticia! El modelo lo clasifica como **Bueno**."
    else: # Regular
        st.warning(f"🟡 Clasificación: {result}")
        color_hex = "#ffbf00ca" # Amarillo
        estrellas = "★★☆"
        mensaje = "Calidad intermedia. El modelo lo clasifica como **Regular**."

    
    st.markdown(
        f"""
        <div style="
            background-color: {color_hex}; 
            padding: 20px; 
            border-radius: 10px; 
            text-align: center;
            color: white;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);">
            <h1 style='color: white; margin: 0;'>{estrellas}</h1>
            <p style='margin: 0;'>{mensaje}</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    st.markdown(f"**Valor de Predicción Continua:** `{prediction_cont:.4f}` (Rango Regular: 0.75 a 0.94)")
    st.markdown("El modelo AdaBoost V2 tuvo un F1-Score Ponderado de **0.73** en esta tarea.")