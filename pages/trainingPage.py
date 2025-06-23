from components.model import VoiceFakeDetection
from fastai.vision.all import *
import streamlit as st

st.title("Computer Vision Model Training for AudioFake Detection")

# Initialize the model
model = VoiceFakeDetection()

# Architecture selection
architecture_name = st.selectbox("🛠️ Choose the architecture", list(model.architectures.keys()))

# Transformation selection
transform_type = st.selectbox("🔄 Choose the transformation", list(model.transforms.keys()))

# Number of epochs 
num_epochs = st.number_input("⏳ Number of Epochs", min_value=0, step=1, value=1)

# Number of batches
num_batches = st.number_input("🧺 Number of Batches", min_value=0, step=2, value=32)

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
        )
        callbacks_values.append(cb_value)
    st.session_state.callbacks = callbacks_values

    st.button("➕ Add Callback", on_click=lambda: st.session_state.callbacks.append(""))
st.session_state.valid_callbacks = st.empty()


default_datasets = {
  "Scottish man": "awb",
  "American man 1": "bdl",
  "American man 2": "rms",
  "American woman 1": "clb",
  "American woman 2": "slt",
  "Canadian man": "jmk",
  "Indian man": "ksp"
} # SPEAKERS from components/VoCoderRecognition/scripts/env.sh 

selected_speakers = [default_datasets[spk] for spk in st.multiselect(
    "🗣️ Choose one or more of our datasets", 
    list(default_datasets.keys())
)]
st.session_state.select_speaker = st.empty()

noise_labels = {
    "Extremely noisy": 10,
    "A lot of noise": 1,
    "Moderate noise": 0.1,
    "Low noise": 0.01,
    "Very low noise": 0.001,
    "No noise": 0
} # NOISES from components/VoCoderRecognition/scripts/env.sh 

selected_noises = [noise_labels[spk] for spk in st.multiselect(
    "🌪️ Choose how much noise you want to train with.", 
    list(noise_labels.keys()),
    default=["No noise"]
)]

def safe_eval_callback(callbacks: list) -> list:
    safe_callbacks = []
    for cb in callbacks:
        cb = cb.strip()
        if cb == "":
            continue

        cb = eval(cb)
        
        # Verifica se é uma instância de Callback
        if not isinstance(cb, Callback):
            raise Exception(f"'{cb}' is not a valid fastai Callback.")

        safe_callbacks.append(cb)

    return safe_callbacks
        
try:
    safe_callbacks = safe_eval_callback(st.session_state.callbacks)
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

