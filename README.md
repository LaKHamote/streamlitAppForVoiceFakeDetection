# 🤖 Fake Voice Detection with FastAI & Streamlit

This project develops and deploys a fake voice detection application using deep learning with FastAI and an interactive interface built with Streamlit.

---

## 📦 Features

- **Transfer Learning Model Training**: Supports training with various transfer learning models (e.g., VGG16, ResNet).
- **Interactive Streamlit Interface**: Provides a user-friendly web interface for interaction.
- **WAV File Upload**: Allows users to upload `.wav` audio files for fake voice detection.
- **Trained Model Download**: Enables downloading of the trained model in `.pkl` format.
- **Dockerized Application**: The entire application is containerized using Docker, ensuring consistent environments across different machines and simplifying deployment. This means you can run the app anywhere Docker is installed, without worrying about dependency conflicts.

---

## 🚀 Requirements

- **Docker**: `version==XXXX`.
- **NVIDIA Toolkit**: For GPU support. Example: `sudo apt install nvidia-container-toolkit`.
- **NVIDIA Docker Runtime**: Configure Docker to use the NVIDIA runtime. Example: `sudo nvidia-ctk runtime configure --runtime=docker` followed by `sudo systemctl restart docker`.
- **Python**: Version 3.10 or higher (optional for local testing).

---

## 📁 Directory Structure

```bash
├── app.py                     → Streamlit application
├── Dockerfile                 → Application container definition
├── requirements.txt           → Project dependencies
├── README.md                  → This README file
└── ...
```

---

## ⚙️ Usage (Linux)

### 1. Clone the Repository (with Submodule)

`git clone --recurse-submodules https://github.com/LaKHamote/streamlitAppForVoiceFakeDetection.git`
cd streamlitAppForVoiceFakeDetection

### 2. Prepare Default Dataset

To prepare the default dataset, execute the following commands:

`pip install -r requirements_min.txt`
`source components/VoCoderRecognition/setup.sh`

This script will perform the following actions:

- Download speakers (defined by `SPEAKERS`) from `components/VoCoderRecognition/scripts/env.sh`.
- Download vocoders (defined by `VOCODER_TAGS`) from `components/VoCoderRecognition/scripts/env.sh`.
- Resample audio frequencies to 22050Hz (required by vocoders).
- Generate new audio for each vocoder.
- Generate Mel Spectrograms for each noise level (defined by `NOISE_LEVEL_LIST`) in `components/VoCoderRecognition/scripts/env.sh`.

If you wish to use your own dataset, please follow the organizational structure of the default dataset and modify the dataset directory to be mounted in `docker-compose.yaml` (e.g., line 11 in the example of yaml below) and change the variables in `components/VoCoderRecognition/scripts/env.sh` accordingly.

```bash
your_dataset/
├── noise1/
│   ├── speaker1/
│   │   ├── class1/
│   │   │   │── image.png
│   │   │   └── ...
│   │   │── class2/
│   │   └── ...
│   │── speaker2/
│   └── ...
│── noise2/
└── ...
```

### 3. Build with Docker Compose

Create a Docker volume to reuse downloaded weights:

Example `docker-compose.yml`:

```yaml
services:
  app:
    build: .
    ports:
      - your_open_port:8501 # Replace 'your_open_port' with your desired port
    volumes:
      - .:/app
      - ./your_dataset:/dataset # You can change the dataset folder mount path to your desired location with the dataset in the correct format
      - ~/.cache/torch/hub:/root/.cache/torch/hub # This prevents re-downloading weights after training
    environment:
      - STREAMLIT_SERVER_PORT=8501
    runtime: nvidia # Enable GPU support`
```

### 4. Start the Application

Execute the following command in your terminal:

Bash

`docker compose up --build`

---

## 🧠 Available Models

You can select from the following deep learning architectures for model training:

- VGG16
- VGG19
- ResNet18
- ResNet34
- ResNet50
- AlexNet

---

## 💾 Training Download Options

After training, the following options will be available:

- View and download the loss history (as a graph and CSV).
- Download the trained model in `.pkl` format.
- Download the confusion matrix.

---

## 🧪 Audio Testing

Once a model is trained or loaded:

1. (Optional) Upload your own model.
2. Upload a `.wav` audio file.
3. View the prediction result.
4. Examine the probabilities for each class.

---

## 📄 License

MIT © Lucas ???
