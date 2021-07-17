import streamlit as st
import pandas as pd
from sklearn import datasets
import pickle

log_model = pickle.load(open('log_model.pkl', 'rb'))
knn_model = pickle.load(open('knn_model.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))

st.write("""
# Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

activities = ['Logistic Regression', 'K Nearest Neighbour', 'Random Forest']
st.sidebar.header('Which model would you like to use?')
option = st.sidebar.selectbox('SELECT MODEL', activities)

st.sidebar.subheader('User Input Parameters')


def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

st.subheader(option)

st.subheader('User Input Parameters')
st.write(df)

iris = datasets.load_iris()

if option == 'Logistic Regression':
    prediction = log_model.predict(df)
    prediction_proba = log_model.predict_proba(df)
elif option == 'K Nearest Neighbour':
    prediction = knn_model.predict(df)
    prediction_proba = knn_model.predict_proba(df)
else:
    prediction = rf_model.predict(df)
    prediction_proba = rf_model.predict_proba(df)

st.subheader('Prediction: Flower Type')
st.write(iris.target_names[prediction])

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction Probability')
st.write(prediction_proba)
