import streamlit as st
import pandas as pd
import pickle
import dill

# ---------- Load the saved schema and the saved pipeline ----------
with open('my_feature_dict.pkl', 'rb') as file:
    my_feature_dict = pickle.load(file)

with open('pipeline.pkl', 'rb') as file:
    model = dill.load(file)


# ---------- Function to predict ----------
def predict_leave(data):
    prediction = model.predict(data)
    return prediction


# ---------- Header and sub-header ----------
st.title('Employee Churn Prediction App')
st.subheader('Created by Abdul Ahad')
st.write('Fill in the employee details below and click Submit to see if the employee is likely to leave.')


# ---------- Categorical inputs (dropdowns) ----------
st.subheader('Categorical Features')

categorical_input = my_feature_dict.get('CATEGORICAL')
categorical_input_vals = {}

for i, col in enumerate(categorical_input.get('Column Name').values()):
    categorical_input_vals[col] = st.selectbox(col, categorical_input.get('Members')[i], key=col)


# ---------- Numerical inputs (number boxes) ----------
st.subheader('Numerical Features')

numerical_input = my_feature_dict.get('NUMERICAL')
numerical_input_vals = {}

for i, col in enumerate(numerical_input.get('Column Name').values()):
    min_value = numerical_input.get('Min')[i]
    max_value = numerical_input.get('Max')[i]
    numerical_input_vals[col] = st.number_input(col, min_value=min_value, max_value=max_value,
                                                value=min_value, step=1, key=col)


# ---------- Combine all inputs into one row of data ----------
input_data = dict(list(categorical_input_vals.items()) + list(numerical_input_vals.items()))
input_data = pd.DataFrame([input_data])


# ---------- Predict when Submit is clicked ----------
if st.button('Submit'):
    prediction = predict_leave(input_data)[0]

    if prediction == 1:
        st.write('The Prediction is **Leave**, Hence the employee is **Expected** to leave the company.')
    else:
        st.write('The Prediction is **Stay**, Hence the employee is **Not Expected** to leave the company.')
