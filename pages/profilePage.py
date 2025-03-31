import streamlit as st
from components.login import Login
import os 
import io
import pandas as pd
from PIL import Image

login = Login()

login.resetPassword()

st.title("Your Models")
st.write("Here you can see all the models you have trained and download it to test them with audios.")
st.info("Notice that this is saved temporarily, so you should download it before finishing your session.")

model_path = f".tmp/{st.session_state.username}"
os.makedirs(model_path, exist_ok=True)
options = st.radio("Choose a model:", [d for d in os.listdir(model_path) for m in os.listdir(f"{model_path}/{d}") if m.endswith(".pkl")])

if options:
    model_buffer = io.BytesIO()
    with open(f"{model_path}/{options}/model.pkl", "rb") as file:
        model_buffer.write(file.read())  
    model_buffer.seek(0)
    st.download_button(
        label="📥 Download Model", 
        data=model_buffer, 
        file_name=f"{options}.pkl",
        mime="application/octet-stream"
    )
    
    st.checkbox("activate")

    col1, col2 = st.columns(2)

    with col1:
        history_data = pd.read_csv(f"{model_path}/{options}/history.csv")
        csv_buffer =  io.BytesIO()
        history_data.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        st.download_button(
            label="📉 Download Training & Validation Loss", 
            data=csv_buffer, 
            file_name="loss_values.csv", 
            mime="text/csv"
        )

    with col2:
        st.subheader("🔢 Confusion Matrix")
        img_buffer = io.BytesIO()
        with open(f"{model_path}/{options}/confusion_matrix.png", "rb") as img_file:
            img_buffer.write(img_file.read())
        img_buffer.seek(0)

        image = Image.open(img_buffer)
        st.image(image, use_container_width=False)

        st.download_button(
            label="📥 Download Confusion Matrix", 
            data=img_buffer, 
            file_name="confusion_matrix.png", 
            mime="image/png"
        )

    st.write("---")




    
