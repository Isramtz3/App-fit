
import numpy as np
import streamlit as st
import pandas as pd
# Importar la librería necesaria para el clasificador (DecisionTreeClassifier ya está importado)
from sklearn.tree import DecisionTreeClassifier

# Configuración inicial de la aplicación
st.set_page_config(page_title="Predicción de Fitness", layout="centered")

st.write(''' # Predicción del estado físico de una persona ''')
# Asegúrate de que tienes un archivo "Manzana.jpg" accesible o usa una URL pública
# Ejemplo de placeholder si no tienes la imagen:
st.image("Manzana.jpg", caption="Analicemos si está en forma una persona en base a sencillos planteamientos.")

st.header('Datos de evaluación')

# Función para la entrada de características del usuario
def user_input_features():
    st.sidebar.header('Parámetros de Entrada')

    # Corrección: Asegúrate de que value >= min_value

    # Entrada de datos numéricos
    Edad = st.sidebar.slider('Edad:', min_value=1, max_value=100, value=30, step=1)
    Altura = st.sidebar.number_input('Altura en cm:', min_value=50, max_value=230, value=170, step=1)
    Peso = st.sidebar.number_input('Peso en kg:', min_value=10.0, max_value=200.0, value=70.0, step=0.1)
    Frecuencia = st.sidebar.number_input('Frecuencia cardíaca en reposo (lpm):', min_value=30.0, max_value=120.0, value=70.0)
    Presion = st.sidebar.number_input('Presión arterial sistólica (mmHg):', min_value=60.0, max_value=180.0, value=120.0)
    Sueño = st.sidebar.number_input('Horas de sueño al día:', min_value=0.0, max_value=12.0, value=7.5)
    
    # Input Quality (min_value=1.0, value debe ser >= 1.0)
    Nutricion = st.sidebar.number_input('Calidad de nutrición (1-10):', min_value=1.0, max_value=10.0, value=5.0)
    Actividad = st.sidebar.number_input('Índice de actividad física (1-5):', min_value=1.0, max_value=5.0, value=3.0)
    
    # Datos categóricos con selectbox para mejor UX
    Fuma_opcion = st.sidebar.selectbox("¿Es fumador activo?", ('No', 'Sí'))
    Genero_opcion = st.sidebar.selectbox("Género", ('Mujer', 'Hombre'))

    # Mapeo a valores numéricos
    Fuma = 1 if Fuma_opcion == 'Sí' else 0
    Genero = 1 if Genero_opcion == 'Hombre' else 0 # 0: mujer, 1: hombre

    # Diccionario de entrada
    user_input_data = {
        'age': Edad, 
        'height_cm': Altura, 
        'weight_kg': Peso, 
        'heart_rate': Frecuencia, 
        'blood_pressure': Presion,
        'sleep_hours': Sueño, 
        'nutrition_quality': Nutricion, 
        'activity_index': Actividad, 
        'smokes': Fuma,
        'gender': Genero
    }

    features = pd.DataFrame(user_input_data, index=[0])
    
    st.subheader('Resumen de los Datos de Entrada')
    st.write(features)

    return features

# Llamada a la función para obtener los datos
df = user_input_features()

# --- MODELO DE CLASIFICACIÓN ---
# Nota: La línea `titanic.drop(columns='is_fit')` y `titanic['is_fit']` parece ser un error de copia.
# Se ha corregido para usar el archivo 'fitness' que mencionas.

try:
    # Carga del dataset (asegúrate de que el archivo 'Fitness_Classification_ok.csv' esté en el mismo directorio)
    fitness_df = pd.read_csv('Fitness_Classification_ok.csv', encoding='latin-1')
  
    X = fitness_df.drop(columns=['is_fit']) # Características de entrenamiento
    Y = fitness_df['is_fit']                # Variable objetivo

    # Entrenamiento del clasificador
    classifier = DecisionTreeClassifier(max_depth=5, criterion='entropy', min_samples_leaf=25, max_features=4, random_state=1615170)
    classifier.fit(X, Y)

    # Predicción
    prediction = classifier.predict(df)

    st.subheader('Predicción')
    if prediction[0] == 0:
        st.error('Basado en los datos, la predicción es que **NO ESTÁ EN FORMA** (is_fit = 0).')
    elif prediction[0] == 1:
        st.success('Basado en los datos, la predicción es que **ESTÁ EN FORMA** (is_fit = 1).')

except FileNotFoundError:
    st.error("Error: No se pudo encontrar el archivo 'Fitness_Classification_ok.csv'. Asegúrate de que esté en la misma ubicación que tu script.")
except KeyError as e:
    st.error(f"Error: La columna {e} no se encuentra en el archivo CSV o hay un error en el nombre de las columnas.")
except Exception as e:
    st.error(f"Ocurrió un error durante el entrenamiento o la predicción: {e}")
