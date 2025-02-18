import streamlit as st
from fastai.vision.all import load_learner, ClassificationInterpretation
import matplotlib.pyplot as plt

st.markdown(st.session_state)


if "uploaded_file" in st.session_state:
    model = load_learner(st.session_state.uploaded_file)