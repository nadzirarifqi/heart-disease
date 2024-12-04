import streamlit as st
from PIL import Image
import overview as ov
import data_preprocessing as pre
import heart as h
import modelling as m
import warnings
warnings.filterwarnings("ignore")


# Side bar
outline = st.sidebar.selectbox('Outline',('Overview', 'Data Preprocessing','Modelling','Predict','About'))

def about():
    st.title('Machine Learning Dashboard')
    st.write('created by : [@nadzirarifqi](https://www.linkedin.com/in/nadzira-rifqi)')
    st.header('My Profile')
    profile, bio = st.columns(2)

    with profile:
        image = Image.open('media/profile.jpg')
        st.image(image)
    
    with bio:
        '''
        Nadzira Rifqi Amin Rinawan \n
        Education:
        Informatics Student at Universitas Sebelas Maret
        '''

if outline == 'About':
    about()
elif outline == 'Overview':
    ov.overview()
elif outline == 'Data Preprocessing':
    pre.data_preprocessing()
elif outline == 'Modelling':
    m.modelling()
elif outline == 'Predict':
    h.heart()


