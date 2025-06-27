import streamlit as st
from components.login import Login

# Application title
st.title("VoiceFake Detection")

Login(False)

st.write("Please choose on of the following options:")

if st.button("Train a model from my dataset", key="to_training"):
    st.switch_page("pages/trainingPage.py")
    
if st.button("Import my own model", key="to_import"):
    st.switch_page("pages/profilePage.py")

