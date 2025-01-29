import streamlit as st
from fastai.vision.all import load_learner


uploaded_file = st.file_uploader("Upload a PKL file containing the model", type=["pkl"])

if uploaded_file is not None:
    model = load_learner(uploaded_file)

    print(model.eval())


from context.userContext import getUserContext
getUserContext()
print(st.session_state)



