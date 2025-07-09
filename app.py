import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Carregando o modelo treinado
@st.cache_resource
def load_model():
    return joblib.load("iris_tree_model.pkl")

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
        'sepal length (cm)': [sepal_length],
        'sepal width (cm)': [sepal_width],
        'petal length (cm)': [petal_length],
        'petal width (cm)': [petal_width]
    })
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    st.subheader("Prediction Results")
    st.write(f"The specie predicted by the model is: **{prediction[0].upper()}**")
    st.write("Prediction probabilities:")
    st.write({
        'Setosa': f"{prediction_proba[0][0]:.2%}",
        'Versicolor': f"{prediction_proba[0][1]:.2%}",
        'Virginica': f"{prediction_proba[0][2]:.2%}"
    })

st.divider() # Divider row to organize the layout

# Random test data
st.header("Second step: Test with random data")
if st.button("Generate Random Data"):
    rand_sl = np.random.uniform(4.0, 8.0)
    rand_sw = np.random.uniform(2.0, 4.5)
    rand_pl = np.random.uniform(1.0, 7.0)
    rand_pw = np.random.uniform(0.1, 2.5)

    # Showing the random data
    st.write("Randomly generated data:")
    st.write(f"Sepal Length: {rand_sl:.1f} cm")
    st.write(f"Sepal Width: {rand_sw:.1f} cm")
    st.write(f"Petal Length: {rand_pl:.1f} cm")
    st.write(f"Petal Width: {rand_pw:.1f} cm")

    # Structuring the random data for prediction
    random_data = pd.DataFrame(
        {
            "sepal length (cm)": [rand_sl],
            "sepal width (cm)": [rand_sw],
            "petal length (cm)": [rand_pl],
            "petal width (cm)": [rand_pw],
        }
    )

    # Prediction
    rd_prediction = model.predict(random_data)

    st.subheader("Random Data Prediction Results")
    st.write(f"The specie predicted by the model for the random data is: **{rd_prediction[0].upper()}**")
