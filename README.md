# 🎙️ Fake Voice Detection with FastAI & Streamlit

Este projeto treina e executa uma aplicação de detecção de voz falsa usando aprendizado profundo com FastAI e uma interface interativa em Streamlit.

## 📦 Funcionalidades

- Treinamento de modelos baseados em Transfer Learning (VGG16, ResNet, etc.)
- Interface interativa com Streamlit
- Upload de arquivos `.wav` para detecção
- Download do modelo treinado (`.pkl`)
- Evita re-download de pesos usando volumes Docker

## 🚀 Requisitos

- Docker version==XXXX
- nvidia toolkit. Ex: sudo apt install nvidia-container-toolkit
- ativar runtime nvida para seu docker: Ex: sudo nvidia-ctk runtime configure --runtime=docker e depois reiniciar sudo systemctl restart docker
- Python3.10 ou maior (para testes locais, opcional)

## 📁 Estrutura de Diretórios

.

├── app.py                 → Aplicativo Streamlit

├── Dockerfile             → Container da aplicação

├── requirements.txt       → Dependências do projeto

├── README.md              → Este arquivo

└── ...

## ⚙️ Como usar (Linux)

### 1. Atualizar submodule

git submodule update --remote --merge

### 1. Preparar o dataset default (precisa de scipy==1.10.1, librosa e parallel_wavegan instaldo)



source components/VoCoderRecognition/setup.sh

O comando vai:
    baixar os locutores(SPEAKERS) em "components/VoCoderRecognition/scripts/env.sh"
    baixar os vocoders(VOCODER_TAGS) em "components/VoCoderRecognition/scripts/env.sh"
    fazer remostragem de frequencia para 22050Hz(exigido pelos vocoders).
    Agora vai gerar um novo audio para cada vocoder
    Finalmente vai gerar um Espectrograma Mel para cada tanto de ruído adicionado (NOISE_LEVEL_LIST) em "components/VoCoderRecognition/scripts/env.sh"

Caso queira usar seu próprio dataset, siga a organizacao do dataset default e altere o diretoria a ser montado no docker-compose.yaml(exemplo linha 11)

-seu_dataset
--ruido1
---locutor1
----classe1
-----image.png
----classe2
---locutor2
--ruido2
...


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

- ./your_personal_dataset:/dataset  # Pode alterar a montagem da pasta dataset para o caminho que quiser com o dataset no formato correto
- ~/.cache/torch/hub:/root/.cache/torch/hub  # Isso evita o redownload dos pesos depois de um treinamento

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

## 💾 Download do Trainamento

Após o treinamento, será possível:

- Ver o histórico de loss e baixá-lo (como gráfico e csv)
- Fazer download do modelo `.pkl`
- Baixar confusion matrix

## 🧪 Testar com Áudio

Após treinar ou carregar um modelo:

0. (Opcional) Faça upload de um modelo seu
1. Faça upload de um arquivo `.wav`
2. Veja a predição
3. Confira as probabilidades de cada classe


## 📄 Licença

MIT © Lucas ??
