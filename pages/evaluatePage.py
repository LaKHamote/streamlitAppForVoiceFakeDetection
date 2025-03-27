import streamlit as st
from fastai.vision.all import load_learner, ClassificationInterpretation
from components.VoCoderRecognition.lib.simple_detector_creator import generate_single_spec
import os
from PIL import Image
from context.userContext import getUserContext
import pandas as pd

def uploaded_model_button():
    new_model = st.file_uploader("Upload a PKL file containing the model", type=["pkl"])
    if new_model:
        global model
        try:
            model = load_learner(new_model)
            show_model_eval = st.checkbox("🔍 Show Model Evaluation")
            if show_model_eval:
                st.write(model.eval())

        except:
            st.error("Error loading model")

getUserContext()
model = None
SAVE_DIR = os.path.join(".tmp", st.session_state.username, "uploads")
os.makedirs(SAVE_DIR, exist_ok=True)


if "trained_model" in st.session_state: # this case, it detected that the user just trained a model
    st.success("It was detected a Model just trained successfully!")

    option = st.radio("Choose an option:", ["Use trained model", "Upload a new model"])

    if option == "Use trained model":
        model = load_learner(st.session_state.trained_model)

    if option == "Upload a new model":
        uploaded_model_button()
        # Load a model
        
else:
    st.warning("It was not detected a Model just trained yet! But you can try upload your own to to test an audio.")
    uploaded_model_button()


if model is not None:
    # Upload an audio file to test the model
    uploaded_audio = st.file_uploader("Upload an audio to test", type=["wav"])

    if uploaded_audio:
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
                            0,
                            crop_width=64,
                            discard_if_too_narrow=True
                        )

        spec_name = os.path.splitext(uploaded_audio.name)[0]+".png"
        st.image(os.path.join(SAVE_DIR, spec_name), use_container_width=False, caption=f"Spectogram of {os.path.splitext(uploaded_audio.name)[0]}")

        test_image = Image.open(os.path.join(SAVE_DIR, spec_name))
        test_image = test_image.resize((128, 128))
        test_image = test_image.convert("RGB")

        pred, pred_idx, probs = model.predict(test_image)

        st.metric(label="🔍 **Predição**", value=f"Classe {pred_idx}: {pred}")
        st.metric(label="🔍 **Probabilidade**", value=probs[pred_idx].item())

        df = pd.DataFrame({
            "Classe": [f"Classe {i}" for i in range(len(probs))],
            "Probabilidade": probs.numpy()
        })
        st.write("📊 **Distribuição das Probabilidades:**")
        st.dataframe(df.style.format({"Probabilidade": "{:.2%}"}), hide_index=True)
        st.bar_chart(df.set_index("Classe"))
