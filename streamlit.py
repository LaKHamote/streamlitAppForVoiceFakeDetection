from model import VoiceFakeDetection


import streamlit as st


def streamlit_interface():
    st.title("Computer Vision Model Training for AudioFake Detection")

    # Initialize the model
    model = VoiceFakeDetection()

    # Architecture selection
    architecture_name = st.selectbox("Choose the architecture", list(model.architectures.keys()))

    # Transformation selection
    transform_type = st.selectbox("Choose the transformation", list(model.transforms.keys()))

    # Number of epochs and batches
    num_epochs = st.number_input("Number of Epochs", min_value=0, step=1)
    num_batches = st.number_input("Number of Batches", min_value=0, step=1)

    # Callbacks
    callbacks = st.text_input(
        "Edit your Callback",
        value="EarlyStoppingCallback(monitor='f1_score', min_delta=0.0001, patience=10)"
    )

    # File upload
    zip_file = st.file_uploader("Upload a ZIP file containing the dataset", type=["zip"])

    # Training button
    if st.button("Train"):
        training_log = model.train_model(architecture_name, transform_type, zip_file, num_epochs, num_batches, callbacks)
        st.text_area("Training Log", value=training_log, height=300)

streamlit_interface()