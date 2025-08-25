import streamlit as st
import numpy as np
import pandas as pd
import pickle
import random
import time

st.header('Heart Disease Prediction Using Machine Learning')

data = '''Heart Disease Prediction using Machine Learning
Heart disease prevention is critical, and data-driven prediction systems can significantly aid in early diagnosis and treatment. Machine Learning offers accurate prediction capabilities, enhancing healthcare outcomes.
In this project, I analyzed a heart disease dataset with appropriate preprocessing. Multiple classification algorithms were implemented in Python using Scikit-learn and Keras to predict the presence of heart disease.

*Algorithms Used*:

- Logistic Regression
- Naive Bayes
- Support Vector Machine (Linear)
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- XGBoost
- Artificial Neural Network (1 Hidden Layer, Keras)'''

st.write(data)

st.image('https://cdn-images-1.medium.com/v2/resize:fit:1600/1*CQXQxHDKi0Q2IpdjhufEcw.jpeg')

with open ('heart_diseases_pred.pkl','rb')as f:
    chatgpt = pickle.load(f)

#Load Data
url = '''https://github.com/ankitmisk/Heart_Disease_Prediction_ML_Model/blob/main/heart.csv?raw=true'''
df = pd.read_csv(url)



st.sidebar.header('Select Features to Predict Heart Disease')
st.sidebar.image('https://media.tenor.com/NRDsqH7bcmgAAAAM/herz-puls.gif')

random.seed(42)

all_values = []

for i in df.iloc[:,:-1]:
    min_value, max_value = df[i].agg(['min','max'])

    var =st.sidebar.slider(f'Select {i} value', int(min_value), int(max_value),
                      random.randint(int(min_value),int(max_value)))

    all_values.append(var)

final_value = [all_values]

ans = chatgpt.predict(final_value)[0]


progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Heart Disease....')
place = st.empty()
place.image('https://i.pinimg.com/originals/00/1c/41/001c41aa841b8d348247b229a961e9a4.gif',width=80)


for i in range(100):
    time.sleep(0.05)
    progress_bar.progress(i + 1)

if ans == 0:
    body = f'No Heart Disease Detected'
    placeholder.empty()
    place.empty()
    st.success(body)
    progress_bar = st.progress(0)
else:
    body = 'Heart Disease Found'
    placeholder.empty()
    place.empty()
    st.warning(body)
    progress_bar = st.progress(0)



