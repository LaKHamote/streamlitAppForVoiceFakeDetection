from components.model import VoiceFakeDetection
from utils.config import load_env_from_sh
from fastai.vision.all import *
import streamlit as st

st.title("Computer Vision Model Training for AudioFake Detection")

# Initialize the model
model = VoiceFakeDetection()

st.header("Model Training Configuration")
col1, col2 = st.columns(2)

with col1:
    # Architecture selection
    architecture_name = st.selectbox(
        "🛠️ Choose the Architecture",
        list(model.architectures.keys()),
        help="Select the Convolutional Neural Network (CNN) architecture for training."
    )

    # Transformation selection
    transform_type = st.selectbox(
        "🔄 Choose the Spectrogram Transformation",
        list(model.transforms.keys()),
        help="Select the type of image transformation to apply to the audio data."
    )

with col2:
    # Number of epochs
    num_epochs = st.number_input(
        "⏳ Number of Epochs",
        min_value=1,
        step=1,
        value=1,
        help="Define the total number of training iterations over the entire dataset." \
            "A higher number can lead to better model performance, but also increases training time."
    )

    # Number of batches
    num_batches = st.number_input(
        "🧺 Batch Size",
        min_value=1,
        step=2,
        value=32,
        help="Specify the number of samples processed before the model's internal parameters are updated." \
            "A larger batch size can speed up training but requires more memory."
    )

######################################
st.header("Dataset Configuration")
default_speakers = {
  "Scottish man": "awb",
  "American man 1": "bdl",
  "American man 2": "rms",
  "American woman 1": "clb",
  "American woman 2": "slt",
  "Canadian man": "jmk",
  "Indian man": "ksp",
} # SPEAKERS from components/VoCoderRecognition/scripts/env.sh 

default_noises = {
    "Extremely noisy": 10,
    "A lot of noise": 1,
    "Moderate noise": 0.1,
    "Low noise": 0.01,
    "Very low noise": 0.001,
    "No noise": 0,
} # NOISE_LEVEL_LIST from components/VoCoderRecognition/scripts/env.sh 

env_speakers, env_noises = load_env_from_sh()

speakers = default_speakers | {
    code: code
    for code in env_speakers
    if code not in default_speakers.values()
} # Merge default speakers with adicional environment speakers

noises = default_noises | {
    code: code
    for code in env_noises
    if code not in default_noises.values()
} # Merge default noises with adicional environment noises


selected_speakers = [speakers[spk] for spk in st.multiselect(
    "🗣️ Choose one or more of our datasets", 
    list(speakers.keys()),
    help="Select the speakers you want to use for training. " \
         "You can choose multiple speakers to create a diverse training dataset.",
)]
st.session_state.select_speaker = st.empty()


selected_noises = [noises[spk] for spk in st.multiselect(
    "🌪️ Choose how much noise you want to train with.", 
    list(noises.keys()),
    default=["No noise"],
    help="Select the noise levels to apply to the audio data during training. " \
         "You can choose multiple noise levels to simulate different conditions."
)]

######################################
st.header("Advanced Configuration")
# Inicializa lista de callbacks
if "callbacks" not in st.session_state:
    st.session_state.callbacks = ["EarlyStoppingCallback(monitor='f1_score', min_delta=0.0001, patience=10)"]

callbacks_values = []
with st.expander("🧩 Callbacks"):
    for i in range(len(st.session_state.callbacks)):
        cb_value = st.text_input(
            label=f"Callback {i+1}",
            value=st.session_state.callbacks[i],
            key=f"callback_input_{i}",
            help="Enter a valid fastai Callback. For example: `EarlyStoppingCallback(monitor='f1_score', min_delta=0.0001, patience=10)`"
        )
        callbacks_values.append(cb_value)
    st.session_state.callbacks = callbacks_values

    st.button("➕ Add Callback", on_click=lambda: st.session_state.callbacks.append(""))
st.session_state.valid_callbacks = st.empty()

        
try:
    safe_callbacks = model.safe_eval_callback(st.session_state.callbacks)
except Exception as e:
    st.session_state.valid_callbacks.error(f"❌ Error in callback: {str(e)}")
    safe_callbacks = None

# Training button
if st.button("🚀 Train"):
    if not selected_speakers:
        st.session_state.select_speaker.warning("⚠️ Please select at least one dataset before training.")
    elif safe_callbacks is not None:
        model.train_model(
            architecture_name,
            transform_type,
            selected_speakers,
            selected_noises,
            num_epochs,
            num_batches,
            safe_callbacks
        )

