import streamlit as st
import pandas as pd

def data_preprocessing():
    st.title("Data Pre-Processing")
    url = "https://storage.googleapis.com/dqlab-dataset/heart_disease.csv"
    column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    data = pd.read_csv(url, names=column_names, skiprows=[0])

    st.header('Data Shape')
    st.write(data.shape)
    st.write('The display above shows the number of rows and columns in the data, namely 1025 patient data and 14 features that help predict heart attack disease.')


    st.header('Data Describe')
    st.write(data.describe())
    st.write('The table above explains the distribution of data on each feature, starting from the number, average, to the maximum  ')
    st.header('Search for Missing Value')
    data.dropna()
    st.write(data.isnull().sum())

    st.header('Data Transformation')
    lst=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'ca']
    data[lst] = data[lst].astype(object)

    # Pelabelan data categorical
    data['sex'] = data['sex'].replace({1: 'Male', 0: 'Female'})
    data['cp'] = data['cp'].replace({0: 'typical angina', 1: 'atypical angina', 2: 'non-anginal pain', 3: 'asymtomatic'})
    data['fbs'] = data['fbs'].replace({0: 'No', 1: 'Yes'})
    data['restecg'] = data['restecg'].replace({0: 'probable or definite left ventricular hypertrophy', 1:'normal', 2: 'ST-T Wave abnormal'})
    data['exang'] = data['exang'].replace({0: 'No', 1: 'Yes'})
    data['slope'] = data['slope'].replace({0: 'downsloping',
                                        1: 'flat',
                                        2: 'upsloping'})
    data['thal'] = data['thal'].replace({1: 'normal',
                                        2: 'fixed defect',
                                        3: 'reversable defect'})
    data['ca'] = data['ca'].replace({0: 'Number of major vessels: 0',
                                    1: 'Number of major vessels: 1',
                                    2: 'Number of major vessels: 2',
                                    3: 'Number of major vessels: 3'})
    data['target'] = data['target'].replace({0: 'No disease',
                                            1: 'Disease'})
    
    data_trans, data_head = st.columns([1,3])
    with data_trans:
        st.write(data.dtypes)
    with data_head:
        st.write(data.head(14))
