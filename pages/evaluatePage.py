import streamlit as st
from fastai.vision.all import load_learner, ClassificationInterpretation
from components.VoCoderRecognition.lib.simple_detector_creator import generate_single_spec
import os
from PIL import Image


SAVE_DIR = os.path.join(".tmp", st.session_state.username, "uploads")
os.makedirs(SAVE_DIR, exist_ok=True)

if "uploaded_file" in st.session_state:
    model = load_learner(st.session_state.uploaded_file)
    uploaded_audio = st.file_uploader("Upload an audio to test", type=["wav"])

    if uploaded_audio is not None:
        st.write(f"Audio uploaded: {uploaded_audio.name}")

        save_path = os.path.join(SAVE_DIR, uploaded_audio.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_audio.read())

        st.audio(save_path, format="audio/wav")

        generate_single_spec(
                            "mel",
                            SAVE_DIR,
                            SAVE_DIR,
                            uploaded_audio.name,
                            0
                        )
        spec_name = os.path.splitext(uploaded_audio.name)[0]+".png"
        st.image(os.path.join(SAVE_DIR, spec_name), use_container_width=False, caption=f"Spectogram of {os.path.splitext(uploaded_audio.name)[0]}")

        test_image = Image.open(os.path.join(SAVE_DIR, spec_name))
        #
        test_image = test_image.resize((128, 128))
        test_image = test_image.convert("RGB")

        pred, pred_idx, probs = model.predict(test_image)
        st.code(f"Predição: {pred}")
        st.code(f"Índice da classe predita: {pred_idx}")
        st.code(f"Probabilidades: {probs}")
