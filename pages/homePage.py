import streamlit as st
from components.login import Login

# Application title
st.title("Welcome to the App")

Login(False)

with open("pages/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.write("Please choose on of the following options:")


if st.button("Import my own model", key="to_import"):
    st.switch_page("pages/importPage.py")

if st.button("Train a model from my dataset", key="to_training"):
    st.switch_page("pages/trainingPage.py")
