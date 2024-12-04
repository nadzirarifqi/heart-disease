import streamlit as st
from PIL import Image

def overview():
    with st.container():
        st.title('Developing a Predictive Model for Early Detection and Prevention of Cardiovascular Diseases')
        image = Image.open('media/overview.jpg')
        st.image(
            image,
            caption = 'Illustration of Cardiovascular Disease (CVDs)'
    )
    # Container problem
    with st.container():
        st.header('Overview')
        st.markdown(
            '''
            <style>
            .justify-text {
                text-align: justify;
            }
            </style>

            <div class="justify-text">
                Cardiovascular diseases (CVDs) are the leading cause of death globally, accounting for 17.9 million deaths annually. Heart disease is primarily caused by hypertension, obesity, and unhealthy lifestyles. Early detection of heart disease is crucial, especially in high-risk groups, to ensure timely treatment and prevention.
                The goal is to develop a predictive model for heart disease based on existing patient data to assist doctors in making precise and accurate diagnoses. The ultimate aim is to enable early treatment of heart disease, thereby reducing the mortality rate associated with it.
            </div>
            ''',
            unsafe_allow_html=True
        )
        st.header('Source Data')
        st.write('[Heart-disease-dataset-byUCI](https://archive.ics.uci.edu/dataset/45/heart+disease)')

