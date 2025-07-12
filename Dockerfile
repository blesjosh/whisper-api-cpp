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

# Clone and build whisper.cpp with detailed branch info
RUN git clone https://github.com/ggerganov/whisper.cpp.git
WORKDIR /app/whisper.cpp
RUN git branch -a
RUN git log -1 --pretty=format:"%h %s" 

# Build with proper flags
RUN mkdir -p build && cd build && cmake .. && cmake --build . --config Release

# List all built binaries to verify what's available
RUN find ./build -type f -executable | grep -v '\.o$'

# Create models directory
RUN mkdir -p models

# Download the model with proper verification
RUN wget -O models/base.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

# Verify model file size (should be >100MB) with verbose output
RUN ls -la models/base.en.bin && \
    file_size=$(stat -c%s "models/base.en.bin") && \
    echo "Model file size: $file_size bytes" && \
    if [ "$file_size" -lt 100000000 ]; then \
        echo "Model file too small, download failed"; \
        exit 1; \
    else \
        echo "Model file size verified: $file_size bytes (>100MB)"; \
    fi

# Copy your app code
WORKDIR /app
COPY main.py /app/

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
