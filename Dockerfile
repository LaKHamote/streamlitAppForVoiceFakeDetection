FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install Python manually
RUN apt-get update && apt-get install -y python3.10 python3-pip


RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app

# Copy requirements.txt first to leverage caching
COPY requirements.txt /app/requirements.txt

# Install dependencies (will use cache if requirements.txt has not changed)
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Now copy the rest of the application
COPY . /app/

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
