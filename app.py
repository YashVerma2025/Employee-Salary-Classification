import pandas as pd
import streamlit as st
import pickle
import numpy as np
from PIL import Image

#load the trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Page configuration 
st.set_page_config(page_title = 'Employee Salary Classification', page_icon = '🏦', layout = 'centered' )
st.title('🏦 Employee Salary Classification')
image = Image.open(r"salary-image.webp") 
resized_image = image.resize((1000, 250))
st.image(resized_image)
st.markdown('Predict whether an employee earns >50k or <=50k based on input features')

# sidebar inputs
st.sidebar.header('Provide Employee Details')

age = st.sidebar.slider('Age', 17, 75)
workclass = st.sidebar.selectbox('Workclass', ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov'])
education = st.sidebar.selectbox('Education', ['HS-grad', 'Some-collage', 'Bachelors', 'Masters', 'Assoc-voc', '11th', 'Assoc-acdm', '10th', '7th-8th', 'Prof-school', '9th', '12th', 'Doctorate'])
marital_status = st.sidebar.selectbox('Marital_status', ['Married-civ-spouse', 'Never-married', 'Divorced', 'separated', 'widowed', 'Married-spouse-absent'])
occupation = st.sidebar.selectbox('Occupation', ['Prof-specialty', 'Exec-managerial', 'Craft-repair', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Tech-support', 'Farming-fishing', 'Protective-serv', 'Priv-house-serve', 'Armed-Forces'])
relationship = st.sidebar.selectbox('Relationship', ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'])
race = st.sidebar.selectbox('Race', ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
gender = st.sidebar.radio('Gender', ['Male', 'Female'])
hours_per_week = st.sidebar.number_input('Hours_per_week', min_value=20, max_value=60, step=2)
native_country = st.sidebar.selectbox('Native_country', ['United-states', 'Mexico', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada', 'India', 'Cuba', 'England', 'China', 'Jamaica', 'El-Salvador', 'Dominican-Republic', 'Italy', 'Japan', 'Poland', 'South', 'Vietnam', 'Columbia', 'Guatemala', 'Haiti', 'Portugal', 'Taiwan', 'Iran', 'Peru', 'Nicaragua', 'Ireland', 'France', 'Greece', 'Hong', 'Cambodia', 'Ecuador', 'Trinadad&Tobago', 'Thailand', 'Outlying-US(Guam-USVI-etc)', 'Laos', 'Scotland', 'Yugoslavia'])

#input data frame
input_df = pd.DataFrame(
    {
        'age':[age],
        'workclass':[workclass],
        'education':[education],
        'marital_status':[marital_status],
        'occupation':[occupation],
        'relationship':[relationship],
        'race':[race],
        'gender':[gender],
        'hours_per_week':[hours_per_week],
        'native_country':[native_country]
    }
)

# Nevigation Tabs On The Page 
tab_titles = ['Individual Prediction', 'Multiple Prediction']
tabs = st.tabs(tab_titles)
with tabs[0]:
    st.write('### Provided Data')
    st.write(input_df)

    #predict button
    if st.button("Predict Individual Salary Class") :
        prediction = pipe.predict(input_df)
        st.success(f"Predicted Salary Class: {prediction[0]}")
with tabs[1]:
    #Batch prediction
    st.markdown('---')
    st.markdown('#### Batch Prediction')
    st.markdown('###### NOTE: Order of columns in csv file should be same as individual prediction and having categories among given')
    uploaded_file = st.file_uploader('Upload a CSV file for batch prediction', type = 'csv')

    if uploaded_file is not None :
        batch_data = pd.read_csv(uploaded_file)
        st.write('uploaded data preview', batch_data.head())
        if st.button("Predict Batch Salary Class") :
            batch_preds = pipe.predict(batch_data)
            batch_data['Predictedclass'] = batch_preds
            st.write('Predictions :')
            st.write(batch_data.head())
            csv = batch_data.to_csv(index = False).encode('utf-8')
            st.download_button('Download Predictions CSV', csv, 'Batch_Prediction.csv')
