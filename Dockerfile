FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    ffmpeg \
    curl \
    git

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Download whisper.cpp
RUN git clone https://github.com/ggerganov/whisper.cpp.git \
    && cd whisper.cpp \
    && make \
    && cp main ../ \
    && mkdir -p ../models

COPY main.py /app/main.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
