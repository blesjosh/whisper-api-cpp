FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Set a working directory
WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    wget \
    cmake \
    build-essential \
    python3 \
    python3-pip

RUN pip3 install fastapi uvicorn python-multipart

# Clone and build whisper.cpp
RUN git clone https://github.com/ggerganov/whisper.cpp.git
WORKDIR /app/whisper.cpp

# Build with proper flags
RUN mkdir -p build && cd build && cmake .. && cmake --build . --config Release

# List built binaries for debugging
RUN find ./build -type f -executable -name "main" -o -name "whisper-cli"

# Create models directory
RUN mkdir -p models

# Download both base and tiny models for redundancy
RUN wget -O models/base.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
RUN wget -O models/tiny.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin

# Verify model file sizes
RUN ls -la models/ && \
    base_size=$(stat -c%s "models/base.en.bin") && \
    tiny_size=$(stat -c%s "models/tiny.en.bin") && \
    echo "Base model size: $base_size bytes" && \
    echo "Tiny model size: $tiny_size bytes" && \
    if [ "$base_size" -lt 100000000 ]; then echo "Base model file too small"; fi && \
    if [ "$tiny_size" -lt 10000000 ]; then echo "Tiny model file too small"; fi

# Copy your app code
WORKDIR /app
COPY main.py /app/

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]