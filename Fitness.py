import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


st.write(''' # Predicción del estado físico de una persona ''')
st.image("Manzana.jpg", caption="Analicemos si está en forma una persona en base a sencillos planteamientos.")

st.header('Datos de evaluación')

def user_input_features():
  # Entrada
  Edad = st.number_input('Edad:', min_value=1, max_value=100, value = 1, step = 1)
  Altura = st.number_input('Altura en cm:', min_value=0, max_value=230, value = 0, step = 1)
  Peso = st.number_input('Peso en kg:', min_value=0.0, max_value=100.0, value = 0.0)
  Frecuencia = st.number_input('Frecuencia cardíaca en reposo:',min_value=0.0, max_value=220.0, value = 0.0)
  Presion = st.number_input('Presión arterial en mmHg:', min_value=0.0, max_value=120.0, value = 0.0)
  Sueño = st.number_input('Horas de sueño al día:', min_value=0.0, max_value=12.0, value = 0.0)
  Nutricion = st.number_input('Calidad de nutrición (1-10):', min_value=1.0, max_value=10.0, value = 1.0)
  Actividad = st.number_input('Calidad de actividad física (1-5):', min_value=1.0, max_value=5.0, value = 1.0)
  Fuma = st.number_input("¿Es fumador activo? (0: no, 1: sí): ", min_value=0, max_value=1, value = 0, step = 1)
  genero = st.number_input("¿Es hombre o mujer? (0: mujer, 1: hombre): ", min_value=0, max_value=1, value=0, step=1)

  # Diccionario de entrada
  user_input_data = {'age': Edad , 
                     'height_cm': Altura, 
                     'weight_kg': Peso, 
                     'heart_rate': Frecuencia, 
                     'blood_pressure': Presion,
                     'sleep_hours': Sueño, 
                     'nutrition_quality': Nutricion, 
                     'activity_index': Actividad, 
                     'smokes': Fuma,
                     'gender': genero}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

fitness =  pd.read_csv('Fitness_Classification_ok.csv', encoding='latin-1')
X = fitness.drop(columns='is_fit')
Y = fitness['is_fit']

classifier = DecisionTreeClassifier(max_depth=5, criterion='entropy', min_samples_leaf=25, max_features=4, random_state=1615170)
classifier.fit(X, Y)

prediction = classifier.predict(df)

st.subheader('Predicción')
if prediction == 0:
  st.write('No está en forma')
elif prediction == 1:
  st.write('Está en forma')
else:
  st.write('Sin predicción')
