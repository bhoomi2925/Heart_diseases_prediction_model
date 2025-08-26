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

st.image('https://i0.wp.com/asianheartinstitute.org/wp-content/uploads/2024/11/Understanding-How-Heart-Disease-Impacts-Your-Body.jpg?fit=1572%2C917&ssl=1')

with open ('heart_diseases_pred.pkl','rb')as f:
    chatgpt = pickle.load(f)

#Load Data
url = '''https://github.com/ankitmisk/Heart_Disease_Prediction_ML_Model/blob/main/heart.csv?raw=true'''
df = pd.read_csv(url)



st.sidebar.header('Select Features to Predict Heart Disease')
st.sidebar.image(''https://cdn.prod.website-files.com/6735d9c156803926ec21b042/6790f85fc90d3c619a64f3ea_Arrhythmia-Mechanism-gif.gif'')

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
place.image('https://content.presentermedia.com/files/animsp/00005000/5747/cardiogram_heart_working_lg_wm.gif',width=80)


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



