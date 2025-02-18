import streamlit as st
from fastai.vision.all import load_learner


uploaded_file = st.file_uploader("Upload a PKL file containing the model", type=["pkl"])

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    try:
        model = load_learner(uploaded_file)
        st.write(model.eval())
    except:
        st.error("Error loading model")
        st.write(uploaded_file)



