FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

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

# Clone and build whisper.cpp in /app/whisper.cpp
RUN git clone https://github.com/ggerganov/whisper.cpp.git
RUN mkdir -p whisper.cpp/build && cd whisper.cpp/build && cmake .. && make

RUN mkdir -p whisper.cpp/models
RUN curl -L -o whisper.cpp/models/base.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/models/ggml-base.en.bin

# Copy your FastAPI code into /app
COPY main.py /app/main.py

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
