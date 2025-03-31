from components.model import VoiceFakeDetection
import streamlit as st

st.title("Computer Vision Model Training for AudioFake Detection")

# Initialize the model
model = VoiceFakeDetection()

# Architecture selection
architecture_name = st.selectbox("Choose the architecture", list(model.architectures.keys()))

# Transformation selection
transform_type = st.selectbox("Choose the transformation", list(model.transforms.keys()))

# Number of epochs and batches
num_epochs = st.number_input("Number of Epochs", min_value=0, step=1, value=1)
num_batches = st.number_input("Number of Batches", min_value=0, step=2, value=32)

# Callbacks
callbacks = st.text_input(
    "Edit your Callback",
    value="EarlyStoppingCallback(monitor='f1_score', min_delta=0.0001, patience=10)",
)

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

# Training button
if st.button("Train"):
    if not selected_speakers:
        st.session_state.select_speaker.warning("⚠️ Please select at least one dataset before training.")
    else:
        model.train_model(
            architecture_name,
            transform_type,
            selected_speakers,
            num_epochs,
            num_batches,
            callbacks
        )
