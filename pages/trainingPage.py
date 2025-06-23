from components.model import VoiceFakeDetection
import streamlit as st

st.title("Computer Vision Model Training for AudioFake Detection")

# Initialize the model
model = VoiceFakeDetection()

# Architecture selection
architecture_name = st.selectbox("Choose the architecture", list(model.architectures.keys()))

# Transformation selection
transform_type = st.selectbox("Choose the transformation", list(model.transforms.keys()))

# Number of epochs 
num_epochs = st.number_input("Number of Epochs", min_value=0, step=1, value=1)

# Number of batches
num_batches = st.number_input("Number of Batches", min_value=0, step=2, value=32)

# Inicializa lista de callbacks
if "callbacks" not in st.session_state:
    st.session_state.callbacks = ["EarlyStoppingCallback(monitor='f1_score', min_delta=0.0001, patience=10)"]
  
# Botão para adicionar novo callback (estilo React)

# Renderizar os inputs
callbacks_values = []
st.subheader("🧩 Callbacks")
for i in range(len(st.session_state.callbacks)):
    cb_value = st.text_input(
        label=f"Callback {i+1}",
        value=st.session_state.callbacks[i],
        key=f"callback_input_{i}",
    )
    callbacks_values.append(cb_value)
st.session_state.callbacks = callbacks_values

st.button("➕ Add Callback", on_click=lambda: st.session_state.callbacks.append(""))

default_datasets = {
  "Scottish man": "awb",
  "American man 1": "bdl",
  "American man 2": "rms",
  "American woman 1": "clb",
  "American woman 2": "slt",
  "Canadian man": "jmk",
  "Indian man": "ksp"
} # SPEAKERS from components/VoCoderRecognition/scripts/env.sh 

st.session_state.select_speaker = st.empty()
selected_speakers = [default_datasets[spk] for spk in st.multiselect(
    "Choose one or more of our datasets", 
    list(default_datasets.keys())
)]

noise_labels = {
    "Extremely noisy": 10,
    "A lot of noise": 1,
    "Moderate noise": 0.1,
    "Low noise": 0.01,
    "Very low noise": 0.001,
    "No noise": 0
} # NOISES from components/VoCoderRecognition/scripts/env.sh 

selected_noises = [noise_labels[spk] for spk in st.multiselect(
    "Choose how much noise you want to train with.", 
    list(noise_labels.keys()),
    default=["No noise"]
)]

# Training button
if st.button("Train"):
    if not selected_speakers:
        st.session_state.select_speaker.warning("⚠️ Please select at least one dataset before training.")
    else:
        model.train_model(
            architecture_name,
            transform_type,
            selected_speakers,
            selected_noises,
            num_epochs,
            num_batches,
            st.session_state.callbacks
        )
