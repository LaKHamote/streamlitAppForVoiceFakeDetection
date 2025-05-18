# 🎙️ Fake Voice Detection with FastAI & Streamlit

Este projeto treina e executa uma aplicação de detecção de voz falsa usando aprendizado profundo com FastAI e uma interface interativa em Streamlit.

## 📦 Funcionalidades

- Treinamento de modelos baseados em Transfer Learning (VGG16, ResNet, etc.)
- Interface interativa com Streamlit
- Upload de arquivos `.wav` para detecção
- Download do modelo treinado (`.pkl`)
- Evita re-download de pesos usando volumes Docker

## 🚀 Requisitos

- Docker instalado
- Python (para testes locais, opcional)

## 📁 Estrutura de Diretórios

.

├── app.py                 → Aplicativo Streamlit

├── Dockerfile             → Container da aplicação

├── requirements.txt       → Dependências do projeto

├── README.md              → Este arquivo

└── ...

## ⚙️ Como usar

### 1. Baixar os pesos dos modelos (localmente)

Execute localmente para baixar os pesos:

Importe as bibliotecas:

from fastai.vision.all import *

Crie os dados fictícios:

dls = ImageDataLoaders.from_empty(size=(224, 224), bs=64, num_workers=0)

Baixe os pesos:

for arch in [vgg16, vgg19, resnet18, resnet34, resnet50, alexnet]:

vision_learner(dls, arch=arch)

### 2. Build com Docker Compose

Crie um volume para reusar os pesos baixados:

Exemplo de docker-compose.yml:

version: "3.8"

services:

app:

build: .

ports:

- "8501:8501"

volumes:

- ~/.cache/torch/hub:/root/.cache/torch/hub

### 3. Suba o app

Execute no terminal:

docker compose up --build

## 🧠 Modelos disponíveis

Você pode escolher entre os seguintes modelos de arquitetura para o treinamento:

- VGG16
- VGG19
- ResNet18
- ResNet34
- ResNet50
- AlexNet

## 💾 Download do Modelo

Após o treinamento, será possível:

- Ver o histórico de loss
- Fazer download do modelo `.pkl`

## 🧪 Testar com Áudio

Após treinar ou carregar um modelo:

1. Faça upload de um arquivo `.wav`
2. Veja a predição
3. Confira as probabilidades de cada classe

## 🛠️ Manutenção

Caso precise limpar os modelos:

rm -rf ~/.cache/torch/hub/checkpoints

## 📄 Licença

MIT © Lucas
