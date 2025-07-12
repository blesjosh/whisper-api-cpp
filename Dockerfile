FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Set a working directory
WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    cmake \
    build-essential \
    python3 \
    python3-pip

RUN pip3 install fastapi uvicorn python-multipart

# Clone and build whisper.cpp
RUN git clone https://github.com/ggerganov/whisper.cpp.git
WORKDIR /app/whisper.cpp/build
RUN cmake .. && make
WORKDIR /app

# Download the model
RUN mkdir -p /app/whisper.cpp/models
RUN curl -L -o /app/whisper.cpp/models/base.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/models/ggml-base.en.bin

# Verify the binary exists (this will fail the build if it doesn't)
RUN ls -la /app/whisper.cpp/build/bin/main && chmod +x /app/whisper.cpp/build/bin/main

# Copy your app code
COPY main.py /app/

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
