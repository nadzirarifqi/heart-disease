import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import time

def heart():
    st.title('User-Predict for Heart Disease')
    st.write('Created by : [@nadzirarifqi](https://www.linkedin.com/in/nadzira-rifqi)')
    st.write("""
    This app predicts **Heart Disease**
    """)
    
    st.sidebar.header('User Input Features:')
    
    def user_input_features():
        st.sidebar.header('Manual Input')
        cp = st.sidebar.slider('Chest pain type', 1, 4, 2)
        if cp == 1.0:
            wcp = "Angina-type chest pain"
        elif cp == 2.0:
            wcp = "Unstable pain-type chest pain"
        elif cp == 3.0:
            wcp = "Severe unstable pain-type chest pain"
        else:
            wcp = "Chest pain that is not related to heart problems"
        st.sidebar.write("Type of chest pain felt by the patient", wcp)
        
        thalach = st.sidebar.slider("Maximum heart rate achieved", 71, 202, 80)
        slope = st.sidebar.slider("ST segment slope on electrocardiogram (ECG)", 0, 2, 1)
        oldpeak = st.sidebar.slider("How much ST segment is decreased or depressed", 0.0, 6.2, 1.0)
        exang = st.sidebar.slider("Exercise induced angina", 0, 1, 1)
        ca = st.sidebar.slider("Number of major vessels", 0, 3, 1)
        thal = st.sidebar.slider("Thalium test result", 1, 3, 1)
        
        sex = st.sidebar.selectbox("Gender", ('Woman', 'Man'))
        if sex == "Woman":
            sex = 0
        else:
            sex = 1 
        
        age = st.sidebar.slider("Age", 29, 77, 30)
        
        data = {'cp': cp,
                'thalach': thalach,
                'slope': slope,
                'oldpeak': oldpeak,
                'exang': exang,
                'ca': ca,
                'thal': thal,
                'sex': sex,
                'age': age}
        
        features = pd.DataFrame(data, index=[0])
        return features
    
    input_df = user_input_features()
    
    # Menambahkan gambar
    img = Image.open("media/heart-disease.jpg")
    st.image(img, width=500)
    
    # Button untuk memulai prediksi
    if st.sidebar.button('Predict!'):
        df = input_df
        st.write(df)
        
        # Memuat model yang telah disimpan
        with open("best_model_heart_disease.pkl", 'rb') as file:  
            loaded_model = pickle.load(file)
        
        # Melakukan prediksi dengan model yang dimuat
        prediction = loaded_model.predict(df)
        result = ['No Heart Disease' if prediction == 0 else 'Yes Heart Disease']
        
        st.subheader('Prediction: ')
        output = str(result[0])
        
        # Menampilkan hasil prediksi setelah penundaan
        with st.spinner('Wait for it...'):
            time.sleep(4)
            st.success(f"Prediction of this app is {output}")
