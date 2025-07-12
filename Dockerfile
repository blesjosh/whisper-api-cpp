FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    cmake \
    build-essential \
    python3 \
    python3-pip

# Install FastAPI & Uvicorn
RUN pip3 install fastapi uvicorn python-multipart

# Clone whisper.cpp
RUN git clone https://github.com/ggerganov/whisper.cpp.git

# Build whisper.cpp
RUN mkdir -p whisper.cpp/build && cd whisper.cpp/build && cmake .. && make

# Download base model
RUN curl -L -o whisper.cpp/models/base.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/models/ggml-base.en.bin

# Copy app code
COPY main.py .

# Expose port
EXPOSE 10000

# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
