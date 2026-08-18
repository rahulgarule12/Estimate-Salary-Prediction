import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

# load the trained model

model=tf.keras.load_model('regression_model.h5')
## load the encoders and scaler

with open('open_encoder_geo.pkl','rb') as file:
    label_encoder_geo=pickle.load(file)
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


##  Streamlit App

st.title('Estimated Salary Prediction')

## user input
geography=st.selectbox('Geography',label_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('credit score')
exited=st.selectbox('Exites',[0,1])
tenure=st.slider('Tenure',1,10)
num_of_products=st.slider('number of Products',1,4)
has_cr_card=st.selectbox('HAs Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'Exited':[exited]
})

geo_encoded=label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=label_encoder_geo.get_feature_names_out(['Geography']))

# cvombine one hot encoded columns with input data
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#Scale the input data
input_data_scaled=scaler.transform(input_data)


## Predict estimated Salary
prediction=model.predict(input_data_scaled)
prediction_salary=prediction[0][0]

st.write(f' Predicted Estimated Salary :${prediction_salary:.2f}')