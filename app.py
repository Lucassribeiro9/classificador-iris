import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Carregando o modelo treinado
@st.cache_resource
def load_model():
    model = joblib.load('iris_tree_model.pkl')
    return model

model = load_model()

st.title("Iris Flower Species Prediction")
st.write("This app predicts the species of Iris flowers based on their features.")

# Input dos dados
st.sidebar.header("First step: Input data manually")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.4)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

if st.sidebar.button("Predict"):
    input_data = pd.DataFrame({
        'sepal_length': [sepal_length],
        'sepal_width': [sepal_width],
        'petal_length': [petal_length],
        'petal_width': [petal_width]
    })
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    st.subheader("Prediction Results")
    st.write(f"The specie predicted by the model is: **{prediction[0].upper()}**")
    st.write("Prediction probabilities:")
    st.write({
        'setosa': f"{prediction_proba[0][0]:.2f}",
        'versicolor': f"{prediction_proba[0][1]:.2f}",
        'virginica': f"{prediction_proba[0][2]:.2f}"
    })
    
st.divider() # Divider row to organize the layout